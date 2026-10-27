import Pkg
Pkg.activate("MusicalExpectation")

include("utils/draw.jl")
include("utils/scale.jl")
include("utils/force.jl")
include("utils/data.jl")

using Gen
using DataFrames
using Distributions
using JSON
using Statistics
using CSV
using .ScaleUtils
using .ForceUtils
using .DataUtils

# --- Model definition, copied verbatim from PMFM.ipynb ---
# (duplicated here rather than factored into utils/ so this script stays self-contained
# and PMFM.ipynb / utils/ don't need to change for this analysis)

C_NOTE = 60
C = create_major_scale(C_NOTE)
G = create_major_scale(55)
D = create_major_scale(62)
A = create_major_scale(57)
E = create_major_scale(64)
B = create_major_scale(59)
F♯ = create_major_scale(66)
G♭ = create_major_scale(66)
D♭ = create_major_scale(61)
A♭ = create_major_scale(56)
E♭ = create_major_scale(63)
B♭ = create_major_scale(58)
F = create_major_scale(65)

const KEY_MAP = Dict(
    "C"  => C,
    "G"  => G,
    "D"  => D,
    "A"  => A,
    "E"  => E,
    "B"  => B,
    "F#" => F♯,
    "Gb" => G♭,
    "Db" => D♭,
    "Ab" => A♭,
    "Eb" => E♭,
    "Bb" => B♭,
    "F"  => F
)

σ_gravity = 0.2
σ_magnetism = 1.5
σ_inertia = 0.01
σ_note = 0.025
w_g, w_m, w_i = 1.2, 0.1, 4.0
α = 0.1
σ_step_base  = 0.6
σ_pitch_obs  = 0.7
γ = 0.25

@gen function melody_kernel(k::Int, state::Tuple{Note,Note}, key::MusicalScale, measure_length::Int=8)
    prev_note, curr_note = state

    Δt ~ categorical(fill(1.0 / measure_length, measure_length))

    gravity   = @trace(normal(calculate_gravity(curr_note, key), σ_gravity), :gravity)
    magnetism = @trace(normal(calculate_magnetism(curr_note, key), σ_magnetism), :magnetism)
    inertia   = @trace(normal(calculate_inertia(prev_note, curr_note, key), σ_inertia), :inertia)

    base_force = w_g*gravity + w_m*magnetism + w_i*inertia

    rhythm_scale = (Δt)^γ
    μ_step = α * rhythm_scale * base_force
    σ_step = σ_step_base * sqrt(Δt)

    step ~ normal(μ_step, σ_step)

    next_pitch = curr_note.value + step
    next_pos   = mod(curr_note.position - 1 + Δt, measure_length) + 1
    obs_pitch ~ normal(next_pitch, σ_pitch_obs)

    next_note = Note(next_pitch, next_pos)
    return (curr_note, next_note)
end

chain = Gen.Unfold(melody_kernel)

@gen function melody(K::Int, key=nothing, init_notes=nothing)
    σ_init = 0.05

    if isnothing(init_notes)
        start_pitch ~ normal(key.notes[1].value, σ_init)
        x₁ = Note(start_pitch, 1)
        init_step ~ normal(0, 1)
        x₂ = Note(x₁.value + init_step, 2)
        init_state = (x₁, x₂)
    else
        init_state = init_notes
    end

    states ~ chain(K - 2, init_state, key)

    notes = Note[init_state[1], init_state[2]]
    for st in states
        push!(notes, st[2])
    end

    return notes
end

MEASURE_LENGTH = 8

function particle_filter(num_particles::Int, num_samples::Int,
                         key::MusicalScale, obs_df::DataFrame,
                         model=melody)

    T = nrow(obs_df)
    @assert T ≥ 2 "Need at least two observed notes"

    obs_notes  = Vector{Note}(obs_df.obs_notes)
    init_state = (obs_notes[1], obs_notes[2])

    init_obs  = Gen.choicemap()
    init_args = (2, key, init_state)

    state = Gen.initialize_particle_filter(model, init_args, init_obs, num_particles)

    for t in 3:T
        Gen.maybe_resample!(state, ess_threshold = num_particles ÷ 4)

        K = t
        k = t - 2

        Δt = mod(obs_df[t, :position] - obs_df[t-1, :position], MEASURE_LENGTH)
        Δt = (Δt == 0) ? MEASURE_LENGTH : Δt

        obs = Gen.choicemap()
        obs[:states => k => :Δt]        = Δt
        obs[:states => k => :obs_pitch] = obs_notes[t].value

        new_args = (K, key, init_state)
        Gen.particle_filter_step!(
            state,
            new_args,
            (UnknownChange(), NoChange(), NoChange()),
            obs
        )
    end

    # Final forecast step
    K_forecast = T + 1
    k_forecast = K_forecast - 2

    last_pos = obs_df[T, :position]
    Δt_f = mod(1 - last_pos, MEASURE_LENGTH)
    Δt_f = (Δt_f == 0) ? MEASURE_LENGTH : Δt_f

    obs_f = Gen.choicemap()
    obs_f[:states => k_forecast => :Δt] = Δt_f

    Gen.particle_filter_step!(
        state,
        (K_forecast, key, init_state),
        (UnknownChange(), NoChange(), NoChange()),
        obs_f
    )

    return Gen.sample_unweighted_traces(state, num_samples)
end

_lse(xs) = (m = maximum(xs); m + log(sum(exp, xs .- m)))

# --- Fragment-in-context PPD queries ---
# For each occurrence (context + fragment prefix), run PMFM's particle filter and record
# the resulting posterior predictive distribution over the next note: its mean, its
# highest-likelihood (mode) particle, its spread, and the log-score it assigns to the
# true continuation.

function query_ppd(query::Dict, n_particles::Int, n_samples::Int)
    key = KEY_MAP[query["key"]]
    notes = Float64.(query["full_prefix"])
    positions = Float64.(query["rhythm_prefix"])
    target_note = Float64(query["ground_truth_continuation"])

    obs_df = make_observations(notes, positions, key)
    pf_traces = particle_filter(n_particles, n_samples, key, obs_df)

    last_notes = Float64[tr[:states][end][2].value for tr in pf_traces]

    scores = Gen.get_score.(pf_traces)
    best_i = argmax(scores)
    mode_pred = pf_traces[best_i][:states][end][2].value

    mean_pred = mean(last_notes)
    sd_pred   = std(last_notes)

    log_score = _lse([Distributions.logpdf(Normal(p, σ_pitch_obs), target_note)
                       for p in last_notes]) - log(length(last_notes))

    return (mean_pred, mode_pred, sd_pred, log_score, last_notes)
end

function main()
    n_particles = 1000
    n_samples   = 500
    n_runs      = 10

    queries = JSON.parsefile("data/fragment_context_queries.json")
    println("Loaded $(length(queries)) fragment-context queries, $n_runs runs")

    runs = Int[]
    occurrence_ids = String[]
    mean_preds = Float64[]
    mode_preds = Float64[]
    sd_preds   = Float64[]
    log_scores = Float64[]
    particles_json = String[]

    start_time = time()
    for run in 1:n_runs
        for (i, query) in enumerate(queries)
            mean_pred, mode_pred, sd_pred, log_score, last_notes = query_ppd(query, n_particles, n_samples)

            push!(runs, run)
            push!(occurrence_ids, query["occurrence_id"])
            push!(mean_preds, mean_pred)
            push!(mode_preds, mode_pred)
            push!(sd_preds, sd_pred)
            push!(log_scores, log_score)
            push!(particles_json, JSON.json(last_notes))
        end
        elapsed = round(time() - start_time, digits=1)
        println("  run $run / $n_runs done  ($(elapsed)s elapsed)")
    end

    df = DataFrame(
        run = runs,
        occurrence_id = occurrence_ids,
        mean_pred = mean_preds,
        mode_pred = mode_preds,
        sd_pred = sd_preds,
        log_score = log_scores,
        particles = particles_json,
    )

    CSV.write("exports/fragment_ppd_results_multirun.csv", df)
    println("Wrote $(nrow(df)) rows to exports/fragment_ppd_results_multirun.csv")
end

main()
