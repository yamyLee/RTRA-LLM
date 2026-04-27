import matplotlib.pyplot as plt
from src.visualization.rendering import animate_ship, animate_static_obstacle

def animate_step(x, y, psi, LOA_own, BOL_own, CPA_own, Xob, Yob, psiob, LOA_ob, BOL_ob, CPA_ob, Risk, Vob, step, l):
    if step % 100 == 0:
        animate_ship(x, y, psi, LOA_own * 5, BOL_own * 5, CPA_own, [0.41, 0, 0.41])
        plt.draw()
        plt.pause(0.1)

    if step % 400 == 0:
        for j in range(len(Xob)):
            obs_col = [0.0, 0.7, 0.0]
            if Risk[j] > 0.75:
                obs_col = [1.0, 0.0, 0.0]
            elif Risk[j] > 0.6:
                obs_col = [1.0, 0.6, 0.0]
            elif Risk[j] > 0.35:
                obs_col = [1.0, 0.9, 0.0]

            if Vob[j] > 0.5:
                if step % 100 == 0:
                    colors = [
                        [0, 0, 1],
                        [1, 0.5, 0],
                        [0, 1, 0],
                    ]
                    animate_ship(Xob[j], Yob[j], psiob[j], LOA_ob[j] * 3, BOL_ob[j] * 3, CPA_ob[j], colors[j])
                    plt.draw()
                    plt.pause(0.1)
            else:
                animate_static_obstacle(Xob[j], Yob[j], CPA_ob[j], obs_col)
