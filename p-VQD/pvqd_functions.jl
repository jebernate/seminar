# Gates

function Rzz(N, i, j, theta)
	return chain(N, [cnot(i, j), put(j => Rz(theta)), cnot(i, j)])
end

# ansatze

function ansatz(N, params) # Here we will consider an array of params

    count = 1

    depth = round(Int, (length(params) - N) / (2*N - 1))

    circ = chain(N)

    for d in 1:depth
        
        for i in 1:N
            push!(circ, put(i => Rx(params[count])))
            count += 1 
        end

        for i in 1:N-1
            push!(circ, Rzz(N, i, i+1, params[count]))
            count += 1
        end
    end

    for i in 1:N
        push!(circ, put(i => Rx(params[count])))
        count += 1
    end

    return circ
end

function ansatz_alternating(N, params)

    depth = round(Int, (length(params) - N)/ (2*N - 1))

    count = 1
    circ = chain(N)

    for d in 1:depth

        if (d+1) % 2 == 0  
            for i in 1:N
                push!(circ, put(i => Rx(params[count])))
                count += 1 
            end
        else
            for i in 1:N
                push!(circ, put(i => Ry(params[count])))
                count += 1 
            end
        end

        for j in 1:N-1
            push!(circ, Rzz(N, j, j+1, params[count]))
            count += 1
        end
    end

    if (depth % 2) == 0
        for i in 1:N
            push!(circ, put(i => Rx(params[count])))
            count += 1
        end
    else
        for i in 1:N
            push!(circ, put(i => Ry(params[count])))
            count += 1
        end
    end

    return circ
end

function trotter_step(N, dt, J, B)

    circ = chain(N)

    for i in 1:N-1
        push!(circ, Rzz(N, i, i+1, 2*dt*J))
    end

    for i in 1:N
        push!(circ, put(i => Rx(2*dt*B)))
    end
   
    return circ
end

function overlap_circuit(N, dt, J, B, params, dparams)

    circ = chain(N)

    push!(circ, (ansatz(N, params)))
    push!(circ, (trotter_step(N, dt, J, B)))
    push!(circ, (ansatz(N, params + dparams)))

    return circ
end

function overlap_circuit_alternating(N, dt, J, B, params, dparams)

    circ = chain(N)

    push!(circ, (ansatz_alternating(N, params)))
    push!(circ, (trotter_step(N, dt, J, B)))
    push!(circ, (ansatz_alternating(N, params + dparams)'))

    return circ
end

function evolution_cirucit(N, alternating, params)
    
    if alternating == true
        circ = ansatz_alternating(N, params)
    else
        circ = ansatz(N, params)
    end

    return circ
end

function global_op(N)

    zero_proj_list = [0.5*(I2+Z) for i in 1:N]
	zero_proj = kron(zero_proj_list...)

	return zero_proj
end

function local_op(N)

    zero_proj_loc = 1/N * sum([kron(N, i => 0.5 *(I2+Z)) for i in 1:N])

    return zero_proj_loc
end

# COST FUNCTION AND GRADIENT

function cost_fn(N, circuit, local_cost)

    initial_state = zero_state(N)

    if local_cost
        h = local_op(N)
    else
        h = global_op(N)
    end

    cost = real.(expect(h, initial_state => circuit))

    return 1-cost
end

function cost_and_gradient(N, circuit, trotter_params, params, dparams, local_cost)

    n_params = length(params)
    gradients = zeros(Float64, n_params)

    dispatch!(circuit, vcat(params, trotter_params, -1*reverse(params + dparams)))
    cost = cost_fn(N, circuit, local_cost)
        
    for i in 1:n_params

        ei = zeros(Float64, n_params)
        ei[i] = pi/2

        dispatch!(circuit, vcat(params, trotter_params, -1*reverse(params + dparams + ei)))
        gradient_i = cost_fn(N, circuit, local_cost)

        dispatch!(circuit, vcat(params, trotter_params, -1*reverse(params + dparams - ei)))
        gradient_i -= cost_fn(N, circuit, local_cost)

        gradients[i] = gradient_i/(2*sin(ei[i]))
    end

    return cost, gradients
end

function pvqd_train(N,
                    dt,
                    steps,
                    J,
                    B,
                    alternating,
                    initial_params,
                    initial_dparams,
                    learning_rate,
                    cost_ths,
                    max_iter,
                    local_cost = true)

    params = copy(initial_params) # Shape (steps + 1, n_params)
    dparams = copy(initial_dparams) # Shape (n_params)

    trotter_params = vcat(2*dt*J*ones(N-1), 2*dt*B*ones(N))
    cost = zeros(Float64, steps)

    if alternating
        println("Using alternating ansatz.")
        circuit = overlap_circuit_alternating(N, dt, J, B, params[1,:], dparams)
    else
        circuit = overlap_circuit(N, dt, J, B, params[1,:], dparams)
    end

    println("Initial params = ", params[1,:])
    println("Initial dparams = ", dparams)
    println("-----------------------------------------------")

    # Optimization loop
    
    println("Training for $steps time steps, and $max_iter max opt steps per time.")
    for t in 1:steps

        cost_t = 1
        steps = 0

        for i in 1:max_iter

            cost_t, grads = cost_and_gradient(N, circuit, trotter_params, params[t,:], dparams, local_cost)
            steps += 1
            dparams = dparams - learning_rate * grads

            if cost_t < cost_ths
                break
            end            
        end

        cost[t] = cost_t
        params[t+1, :] = params[t, :] .+ dparams

        @printf("t = %.2f, step infidel after %i steps is %.2e \n", t*dt, steps, cost_t)
    end
    println("-----------------------------------------------")

    return params, cost

end

# Create magnetization operators

function S_xyz(N)

    S_x = 1/N * sum([kron(N, i => X) for i in 1:N])
    S_y = 1/N * sum([kron(N, i => Y) for i in 1:N])
    S_z = 1/N * sum([kron(N, i => Z) for i in 1:N])

    return S_x, S_y, S_z
end

function measure_observable(N, obs, alternating, params, steps)

    initial_state = zero_state(N)

    obs_measured = zeros(Float64, steps + 1)
    circuit = evolution_cirucit(N, alternating, params[1,:])

    for t in 1:(steps+1)

        dispatch!(circuit, params[t,:])
        obs_measured[t] = real(expect(obs, initial_state => circuit))
    end

    return obs_measured
end

# Exact evolution

function create_hamiltonian(N, J, B)

    Hx = sum([kron(N, i => X) for i in 1:N])
    Hzz = sum([kron(N, i => Z, i+1 => Z) for i in 1:N-1])
    H = J*Hzz + B*Hx

    return H 
end

function measure_observables_exact(H, dt, obs, steps)

    obs_measured = zeros(Float64, steps + 1)
    initial_state = zero_state(N)
    
    println("Calculating observables...", obs)

    for i in 1:steps+1
        circ = time_evolve(H, dt*(i-1))
        obs_measured[i] = real(expect(obs, initial_state => circ))
    end

    return obs_measured
end

function calculate_infidelities(N, params, dt, steps)

    fidelities = zeros(Float64, steps + 1)
    initial_state = zero_state(N)

    H = create_hamiltonian(N, J, B)
    circuit = evolution_cirucit(N, alternating, params[1,:])
    println("Calculating infidelities...")

    for t in 1:steps+1

        dispatch!(circuit, params[t,:])
        fidelities[t] = fidelity(initial_state => circuit,  initial_state => time_evolve(H, dt*(t-1)))

    end

    return 1.0 .- fidelities
end