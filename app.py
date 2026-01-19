import streamlit as st
import pandas as pd
from pathlib import Path

from utils.barrel_over_time_utils import generate_bat_path, plot_bat_path

# Get the data directory
data_dir = Path(__file__).resolve().parent / "data"

# Load batter decoder for name -> MLBAMID lookup
@st.cache_data
def load_batter_decoder():
    df = pd.read_csv(data_dir / "batter_decoder.csv")
    return df

batter_decoder = load_batter_decoder()

# Create a dictionary mapping Name -> MLBAMID
name_to_mlbamid = dict(zip(batter_decoder["Name"], batter_decoder["MLBAMID"]))

# Get sorted list of player names
player_names = sorted(batter_decoder["Name"].tolist())

# App title
st.title("Barrel Over Time Visualization")

# Sidebar inputs
st.sidebar.header("Input Parameters")

# Player name dropdown
player_name = st.sidebar.selectbox(
    "Player Name",
    options=player_names,
    index=player_names.index("Shohei Ohtani") if "Shohei Ohtani" in player_names else 0
)

# Get MLBAMID from selected player name
batter = name_to_mlbamid[player_name]

# Count dropdown
count_options = ["other", "batter_ahead", "full", "two_strikes"]
count = st.sidebar.selectbox(
    "Count",
    options=count_options,
    index=0
)

# Float inputs for other parameters
attack_direction_slope_multiplier = st.sidebar.number_input(
    "Attack Direction Slope Multiplier",
    value=1.0,
    min_value=0.0,
    max_value=5.0,
    step=0.1,
    format="%.1f"
)

attack_angle_slope_multiplier = st.sidebar.number_input(
    "Attack Angle Slope Multiplier",
    value=1.0,
    min_value=0.0,
    max_value=5.0,
    step=0.1,
    format="%.1f"
)

pitch_speed = st.sidebar.number_input(
    "Pitch Speed (mph)",
    value=95.0,
    min_value=60.0,
    max_value=110.0,
    step=1.0,
    format="%.1f"
)

init_plate_x = st.sidebar.number_input(
    "Initial Plate X (inches)",
    value=-8.5,
    min_value=-20.0,
    max_value=20.0,
    step=0.5,
    format="%.1f"
)

init_int_y = st.sidebar.number_input(
    "Initial Int Y (inches)",
    value=12.0,
    min_value=0.0,
    max_value=30.0,
    step=0.5,
    format="%.1f"
)

final_x_intercept = st.sidebar.number_input(
    "Final X Intercept (inches)",
    value=8.5,
    min_value=-20.0,
    max_value=20.0,
    step=0.5,
    format="%.1f"
)

# Display selected player info
st.write(f"**Selected Player:** {player_name} (MLBAMID: {batter})")

# Generate and display the plot
if st.button("Generate Plot"):
    with st.spinner("Generating bat path..."):
        try:
            # Generate bat path data
            t_list, plate_x_list, int_y_list, bat_speed_list = generate_bat_path(
                player_name,
                batter,
                count,
                attack_direction_slope_multiplier,
                attack_angle_slope_multiplier,
                pitch_speed,
                init_plate_x,
                init_int_y,
                final_x_intercept,
            )

            # Create the plot using matplotlib
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            from matplotlib.patches import Polygon
            import seaborn as sns
            import numpy as np

            fig, ax = plt.subplots(figsize=(10, 10))

            max_plate = 19

            # Draw home plate
            home_plate = Polygon([
                (-0.7083*12, 0),
                (0.7083*12, 0),
                (0.7083*12, -0.7083*12),
                (0.0, -1.4167*12),
                (-0.7083*12, -0.7083*12),
            ], closed=True, edgecolor="black", facecolor="none", linewidth=2)

            ax.add_patch(home_plate)

            # Filter data for plotting
            mask = [plate_x < max_plate for plate_x in plate_x_list]
            plate_x_filtered = pd.Series(plate_x_list).loc[mask]
            int_y_filtered = pd.Series(int_y_list).loc[mask]
            bat_speed_filtered = pd.Series(bat_speed_list).loc[mask]

            # Plot black background points
            sns.scatterplot(
                ax=ax,
                x=plate_x_filtered,
                y=int_y_filtered,
                color="black",
                s=70,
                edgecolor=None,
                legend=None,
                zorder=0,
            )

            # Plot colored points
            sns.scatterplot(
                ax=ax,
                x=plate_x_filtered,
                y=int_y_filtered,
                s=50,
                hue=bat_speed_filtered,
                palette="Reds",
                edgecolor=None,
                legend=None,
            )

            ax.set_aspect("equal", adjustable="box")

            # Add colorbar
            norm = plt.Normalize(bat_speed_filtered.min(), bat_speed_filtered.max())
            sm = plt.cm.ScalarMappable(cmap="Reds", norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, label="Bat Speed (mph)")

            ax.set(
                xlim=(-18, 18),
                xlabel="",
                ylabel="",
                xticks=[],
            )
            ax.set_ylim(bottom=-18)

            ax.spines[["top", "left", "right", "bottom"]].set_visible(False)

            # Add annotation for pitch speed
            ax.annotate(
                xy=(pd.Series(plate_x_list).iloc[0], pd.Series(int_y_list).iloc[0]),
                xytext=(pd.Series(plate_x_list).iloc[0]-1, pd.Series(int_y_list).iloc[0]+4),
                ha="right",
                text=f"{pitch_speed}mph",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0, connectionstyle="arc3,rad=-0.3"),
            )

            # Add annotation for outer intercept
            t_series = pd.Series(t_list)
            outer_mask = [plate_x < final_x_intercept for plate_x in plate_x_list]
            outer_tf = t_series.loc[outer_mask].max()
            outer_int_y = pd.Series(int_y_list).loc[outer_mask].max()
            outer_mph = 95 - 20 / (0.488 - 0.396) * (outer_tf - 0.396)

            ax.annotate(
                xy=(final_x_intercept, outer_int_y),
                xytext=(final_x_intercept-2, outer_int_y+4),
                ha="center",
                text=f"{outer_mph:.1f}mph pitch\nintercept point",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0, connectionstyle="arc3,rad=-0.3"),
            )

            ax.set_title(label=f"{player_name} Barrel over Time", y=1.05)

            # Format y-axis
            yticklabels = [f"{tick:.1f}" for tick in ax.get_yticks()[1:-1]]
            if yticklabels:
                yticklabels[-1] = yticklabels[-1] + "''"
            ax.set_yticks(ax.get_yticks()[1:-1])
            ax.set_yticklabels(yticklabels)
            ax.tick_params(axis="y", width=0)

            st.pyplot(fig)
            plt.close()

        except Exception as e:
            st.error(f"Error generating plot: {str(e)}")
            st.info("This player may not have sufficient data in the model. Try selecting a different player.")
