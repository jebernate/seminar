import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    N = 7
    depth = 5
    dt = 0.05
    steps = 40

    filename = "data/pvqd_"+str(N)+"_spins_"+str(depth)+"_depth_"+str(dt)+"_dt_"+str(steps)+"_steps.dat"
    f = open(filename, "r")

    data = json.load(f)

    params = data["params"]
    cost = data["cost"]
    ths = data["ths"]
    S_x = data["S_x"]
    S_y = data["S_y"]
    S_z = data["S_z"]
    S_x_exact = data["S_x_exact"]
    S_y_exact = data["S_y_exact"]
    S_z_exact = data["S_z_exact"]
    infidelities = data["infidelities"]

    t = [dt*i for i in range(steps + 1)]
    
    # Cost function and infidelity
    
    plt.title(r"$N$ = {:} spins, $dt$ = {:}, depth = {:}".format(N, dt, depth))
    plt.xlabel(r"$t$")
    plt.ylabel("Infidelity")
    
    plt.semilogy(t[1:], infidelities[1:], 'bs',label=r"$1-|\langle \psi(t) | C(w)\rangle|^2$")
    plt.semilogy(t[1:], cost, 'ko', label=r"$1-|\langle 0| C^{\dagger}(w)e^{iHt}C(w+dw)|0\rangle|^2$")
    plt.axhline(ths, color='k', linestyle = '--')
    
    plt.legend()
    plt.savefig(
        f"figures/fig_cost_{N}_spins_{depth}_depth_{dt}_dt_{steps}_steps.jpg",
        dpi=800,
        bbox_inches="tight"
        )
    plt.clf()


    # Magnetization
    
    fig, ax1 = plt.subplots()
    
    plt.title(r"$N$ = {:} spins, $dt$ = {:}, depth = {:}".format(N, dt, depth))
    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$\langle S \rangle$")
    
    ax1.plot(t, S_x,"b.",label=r"$\langle S_x \rangle$, p-VQD, state sim")
    ax1.plot(t, S_x_exact, "b-", label="Exact")
    
    ax1.plot(t, S_y,"r.",label=r"$\langle S_y \rangle$")
    ax1.plot(t, S_y_exact, "r-")
    
    ax1.plot(t, S_z,"k.",label=r"$\langle S_z \rangle$")
    ax1.plot(t, S_z_exact, "k-")

    ax1.legend()
    fig.savefig(
        f'figures/fig_{N}_spins_{depth}_depth_{dt}_dt_{steps}_steps.jpg',
        dpi=800,
        bbox_inches="tight"
        )
