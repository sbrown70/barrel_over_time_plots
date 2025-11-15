def get_attack_direction_factors(
    batter=660271,
    count="other",
    attack_direction_slope_multiplier=1.5,
):
    from pathlib import Path

    utils_dir = Path(__file__).resolve().parent
    data_dir = utils_dir.parent / "data"

    import polars as pl
    import pandas as pd
    import numpy as np

    attack_direction_summary=pl.read_csv(data_dir / "attack_direction_summary.csv")
    
    base_intercept=attack_direction_summary.filter(pl.col("index")=="Intercept")["mean"].to_numpy()[0]
    
    batter_base_intercept = (
        attack_direction_summary
        .filter(pl.col("index").str.contains(f"{batter}"))
        .filter(~pl.col("index").str.contains("count"))
        .filter(~pl.col("index").str.contains("int_y"))
        ["mean"].to_numpy()[0]
    )
    
    batter_count_intercept = (
        attack_direction_summary
        .filter(pl.col("index").str.contains(f"{batter}"))
        .filter(pl.col("index").str.contains(count))
        .filter(~pl.col("index").str.contains("int_y"))
        ["mean"].to_numpy()[0]
    )
    
    int_y_vs_attack_direction_intercept = base_intercept + batter_base_intercept + batter_count_intercept
    
    batter_base_slope = (
        attack_direction_summary
        .filter(pl.col("index").str.contains(f"{batter}"))
        .filter(~pl.col("index").str.contains("count"))
        .filter(pl.col("index").str.contains("int_y"))
        ["mean"].to_numpy()[0]
    )
    
    batter_count_slope = (
        attack_direction_summary
        .filter(pl.col("index").str.contains(f"{batter}"))
        .filter(pl.col("index").str.contains(count))
        .filter(pl.col("index").str.contains("int_y"))
        ["mean"].to_numpy()[0]
    )
    
    int_y_vs_attack_direction_slope = batter_base_slope + batter_count_slope
    
    int_y_vs_attack_direction_slope=int_y_vs_attack_direction_slope*attack_direction_slope_multiplier
    
    return int_y_vs_attack_direction_intercept, int_y_vs_attack_direction_slope

def get_attack_angle_factors(
    batter=660271,
    count="other",
    attack_angle_slope_multiplier=1.5,
):
    from pathlib import Path

    utils_dir = Path(__file__).resolve().parent
    data_dir = utils_dir.parent / "data"

    import polars as pl
    import pandas as pd
    import numpy as np
    attack_angle_summary=pl.read_csv(data_dir / "attack_angle_summary.csv")
    
    base_intercept=attack_angle_summary.filter(pl.col("index")=="Intercept")["mean"].to_numpy()[0]
    
    batter_base_intercept = (
        attack_angle_summary
        .filter(pl.col("index").str.contains(f"{batter}"))
        .filter(~pl.col("index").str.contains("count"))
        .filter(~pl.col("index").str.contains("int_y"))
        ["mean"].to_numpy()[0]
    )
    
    batter_count_intercept = (
        attack_angle_summary
        .filter(pl.col("index").str.contains(f"{batter}"))
        .filter(pl.col("index").str.contains(count))
        .filter(~pl.col("index").str.contains("int_y"))
        ["mean"].to_numpy()[0]
    )
    
    int_y_vs_attack_angle_intercept = base_intercept + batter_base_intercept + batter_count_intercept
    
    batter_base_slope = (
        attack_angle_summary
        .filter(pl.col("index").str.contains(f"{batter}"))
        .filter(~pl.col("index").str.contains("count"))
        .filter(pl.col("index").str.contains("int_y"))
        ["mean"].to_numpy()[0]
    )
    
    batter_count_slope = (
        attack_angle_summary
        .filter(pl.col("index").str.contains(f"{batter}"))
        .filter(pl.col("index").str.contains(count))
        .filter(pl.col("index").str.contains("int_y"))
        ["mean"].to_numpy()[0]
    )
    
    int_y_vs_attack_angle_slope = batter_base_slope + batter_count_slope
    
    int_y_vs_attack_angle_slope=int_y_vs_attack_angle_slope*attack_angle_slope_multiplier
    
    return int_y_vs_attack_angle_intercept, int_y_vs_attack_angle_slope

def get_bat_speed_factors(
    batter=660271,
    count="other",
):
    
    from pathlib import Path

    utils_dir = Path(__file__).resolve().parent
    data_dir = utils_dir.parent / "data"

    import polars as pl
    import pandas as pd
    import numpy as np
    bat_speed_summary=pl.read_csv(data_dir / "bat_speed_summary.csv")
    
    base_intercept=bat_speed_summary.filter(pl.col("index")=="Intercept")["mean"].to_numpy()[0]
    
    batter_base_intercept = (
        bat_speed_summary
        .filter(pl.col("index").str.contains(f"{batter}"))
        .filter(~pl.col("index").str.contains("count"))
        .filter(~pl.col("index").str.contains("int_y"))
        ["mean"].to_numpy()[0]
    )
    
    batter_count_intercept = (
        bat_speed_summary
        .filter(pl.col("index").str.contains(f"{batter}"))
        .filter(pl.col("index").str.contains(count))
        .filter(~pl.col("index").str.contains("int_y"))
        ["mean"].to_numpy()[0]
    )
    
    int_y_vs_bat_speed_intercept = base_intercept + batter_base_intercept + batter_count_intercept
    
    batter_base_slope = (
        bat_speed_summary
        .filter(pl.col("index").str.contains(f"{batter}"))
        .filter(~pl.col("index").str.contains("count"))
        .filter(pl.col("index").str.contains("int_y"))
        .filter(~pl.col("index").str.contains("squared"))
        ["mean"].to_numpy()[0]
    )
    
    batter_count_slope = (
        bat_speed_summary
        .filter(pl.col("index").str.contains(f"{batter}"))
        .filter(pl.col("index").str.contains(count))
        .filter(pl.col("index").str.contains("int_y"))
        .filter(~pl.col("index").str.contains("squared"))
        ["mean"].to_numpy()[0]
    )
    
    int_y_vs_bat_speed_slope = batter_base_slope + batter_count_slope

    batter_base_slope = (
        bat_speed_summary
        .filter(pl.col("index").str.contains(f"{batter}"))
        .filter(~pl.col("index").str.contains("count"))
        .filter(pl.col("index").str.contains("int_y"))
        .filter(pl.col("index").str.contains("squared"))
        ["mean"].to_numpy()[0]
    )
    
    batter_count_slope = (
        bat_speed_summary
        .filter(pl.col("index").str.contains(f"{batter}"))
        .filter(pl.col("index").str.contains(count))
        .filter(pl.col("index").str.contains("int_y"))
        .filter(pl.col("index").str.contains("squared"))
        ["mean"].to_numpy()[0]
    )
    
    int_y_squared_vs_bat_speed_slope = batter_base_slope + batter_count_slope
    
    return int_y_vs_bat_speed_intercept, int_y_vs_bat_speed_slope, int_y_squared_vs_bat_speed_slope

def generate_bat_path(
    player_name = "Shohei Ohtani",
    batter = 660271,
    count = "other",
    attack_direction_slope_multiplier = 1.5,
    attack_angle_slope_multiplier = 1.5,
    pitch_speed = 95,
    init_plate_x = -8.5,
    init_int_y = 12,
    final_x_intercept = 8.5,
):
    import polars as pl
    import pandas as pd
    import numpy as np

    init_tf = 0.833 -0.0046 * pitch_speed
    
    bat_speed_intercept, bat_speed_int_y_slope, bat_speed_int_y_squared_slope = get_bat_speed_factors(
        batter,
        count,
    )
    
    int_y_vs_attack_direction_intercept, int_y_vs_attack_direction_slope = get_attack_direction_factors(
        batter,
        count,
        attack_direction_slope_multiplier,
    )
    
    int_y_vs_attack_angle_intercept, int_y_vs_attack_angle_slope = get_attack_angle_factors(
        batter,
        count,
        attack_angle_slope_multiplier,
    )
    
    t0=init_tf
    
    t_list=[t0]
    plate_x_list=[init_plate_x]
    int_y_list=[init_int_y-24]
    bat_speed_list=[(bat_speed_intercept+bat_speed_int_y_slope*init_int_y+bat_speed_int_y_squared_slope*(init_int_y**2))]
    
    t=t0
    plate_x=init_plate_x
    int_y=init_int_y
    
    time_step=.00001
    
    while t<(0.6+time_step):
        t=t+time_step
        
        bat_speed = (bat_speed_intercept+bat_speed_int_y_slope*int_y+bat_speed_int_y_squared_slope*(int_y**2))*17.6
        attack_angle = int_y_vs_attack_angle_intercept+int_y_vs_attack_angle_slope*int_y
        attack_direction = int_y_vs_attack_direction_intercept+int_y_vs_attack_direction_slope*int_y
    
        theta = attack_direction*np.pi/180
        phi = attack_angle*np.pi/180
    
        bat_speed_x = np.cos(phi)*np.sin(theta)*bat_speed
        bat_speed_y = np.cos(phi)*np.cos(theta)*bat_speed
    
        plate_x-=(time_step*bat_speed_x)
        int_y+=(time_step*bat_speed_y)
    
        t_list.append(t)
        plate_x_list.append(plate_x)
        int_y_list.append(int_y-24)
        bat_speed_list.append(bat_speed/17.6)

    return t_list, plate_x_list, int_y_list, bat_speed_list

def plot_bat_path(
    player_name,
    batter,
    count,
    attack_direction_slope_multiplier,
    attack_angle_slope_multiplier,
    pitch_speed,
    init_plate_x,
    init_int_y,
    final_x_intercept,
    t_list, 
    plate_x_list, 
    int_y_list, 
    bat_speed_list,
):
    import polars as pl
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig,ax=plt.subplots()
    
    max_plate=19
    
    import matplotlib.patches as patches
    from matplotlib.patches import Polygon
    
    home_plate = Polygon([
        (-0.7083*12, 0),  # top left corner
        (0.7083*12, 0),   # top right corner
        (0.7083*12, -0.7083*12),   # bottom right point
        (0.0, -1.4167*12),         # bottom tip (toward batter)
        (-0.7083*12, -0.7083*12),  # bottom left point
    ], closed=True, edgecolor="black", facecolor="none", linewidth=2)
    
    ax.add_patch(home_plate)
    
    sns.scatterplot(
        ax=ax,
        x=pd.Series(plate_x_list).loc[[plate_x<max_plate for plate_x in plate_x_list]],
        y=pd.Series(int_y_list).loc[[plate_x<max_plate for plate_x in plate_x_list]],
        s=50,
        hue=pd.Series(bat_speed_list).loc[[plate_x<max_plate for plate_x in plate_x_list]],
        palette="Reds",
        edgecolor=None,
        legend=None,
    )
    
    sns.scatterplot(
        ax=ax,
        x=pd.Series(plate_x_list).loc[[plate_x<max_plate for plate_x in plate_x_list]],
        y=pd.Series(int_y_list).loc[[plate_x<max_plate for plate_x in plate_x_list]],
        color="black",
        s=70,
        edgecolor=None,
        legend=None,
        zorder=0,
    )
    
    ax.set_aspect("equal", adjustable="box")  # Equal scaling for both axes
    
    norm = plt.Normalize(
        pd.Series(bat_speed_list).loc[[plate_x<max_plate for plate_x in plate_x_list]].min(),
        pd.Series(bat_speed_list).loc[[plate_x<max_plate for plate_x in plate_x_list]].max()
    )
    sm = plt.cm.ScalarMappable(cmap="Reds", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label="Bat Speed (mph)")
    
    ax.set(
        # ylim=(-1.5*12,40),
        xlim=(-18,18),
        xlabel="",
        ylabel="",
        xticks=[],
        # yticks=[],
    )

    ax.set_ylim(bottom=-18)
    
    ax.spines[["top","left","right","bottom"]].set_visible(False)
    
    ax.annotate(
        xy=(pd.Series(plate_x_list).iloc[0],pd.Series(int_y_list).iloc[0],),
        xytext=(pd.Series(plate_x_list).iloc[0]-1,pd.Series(int_y_list).iloc[0]+4),
        ha="right",
        text=f"{pitch_speed}mph",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0,connectionstyle="arc3,rad=-0.3",),
    )
    
    outer_tf = pd.Series(t_list).loc[[plate_x<final_x_intercept for plate_x in plate_x_list]].max()
    outer_int_y = pd.Series(int_y_list).loc[[plate_x<final_x_intercept for plate_x in plate_x_list]].max()
    
    outer_mph=95-20/(0.488-0.396)*(outer_tf-0.396)
    
    ax.annotate(
        xy=(final_x_intercept,outer_int_y),
        xytext=(final_x_intercept-2,outer_int_y+4),
        ha="center",
        text=f"{outer_mph:.1f}mph pitch\nintercept point",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.0,connectionstyle="arc3,rad=-0.3",),
    )
    
    ax.set_title(
        label=f"{player_name} Barrel over Time",y=1.05,
    )

    yticklabels=[f"{tick:.1f}" for tick in ax.get_yticks()[1:-1]]
    yticklabels[-1] = yticklabels[-1]+"""''"""

    ax.set_yticks(ax.get_yticks()[1:-1])
    ax.set_yticklabels(yticklabels)

    ax.tick_params(axis="y",width=0)
    
    plt.show()

def bat_path_over_time_full(
    player_name = "Shohei Ohtani",
    batter = 660271,
    count = "other",
    attack_direction_slope_multiplier = 1.5,
    attack_angle_slope_multiplier = 1.5,
    pitch_speed = 95,
    init_plate_x = -8.5,
    init_int_y = 12,
    final_x_intercept = 8.5,
):
    
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
    
    plot_bat_path(
        player_name,
        batter,
        count,
        attack_direction_slope_multiplier,
        attack_angle_slope_multiplier,
        pitch_speed,
        init_plate_x,
        init_int_y,
        final_x_intercept,
        t_list, 
        plate_x_list, 
        int_y_list, 
        bat_speed_list,
    )