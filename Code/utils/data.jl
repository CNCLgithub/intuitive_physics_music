module DataUtils

using DataFrames
using CSV
using ..ScaleUtils
using ..ForceUtils

export make_observations, load_rhythms, load_melodies, load_measures

function make_observations(df::DataFrame, idx::Int64)
    melody_sequence = df[!, idx]
    return melody_sequence
end

function make_observations(
    notes::Vector{Float64},
    positions::Vector{Float64},
    key::MusicalScale
)
    @assert length(notes) == length(positions)

    observed_notes = [Note(n, p) for (n, p) in zip(notes, positions)]

    gravity   = calculate_gravity.(observed_notes, Ref(key))
    magnetism = calculate_magnetism.(observed_notes, Ref(key))

    # Second-order inertia
    inertia = Float64[]
    push!(inertia, 0.0)

    for i in 2:length(observed_notes)
        prev_note = observed_notes[i-1]
        curr_note = observed_notes[i]
        push!(
            inertia,
            calculate_inertia(prev_note, curr_note, key)
        )
    end

    return DataFrame(
        obs_notes = observed_notes,
        position  = positions,
        gravity   = gravity,
        magnetism = magnetism,
        inertia   = inertia
    )
end

function load_melodies(path::String)
    df = DataFrame(CSV.File(path, missingstring="NA"))

    melodies    = Vector{Vector{Float64}}()
    human_means = Float64[]
    stemIDs     = String[]

    for col in names(df)[1:end]
        vals = Float64[]

        for v in df[!, col]
            if ismissing(v)
                continue
            elseif v isa Number
                push!(vals, Float64(v))
            elseif v isa AbstractString
                parsed = tryparse(Float64, v)
                parsed === nothing || push!(vals, parsed)
            end
        end

        @assert length(vals) ≥ 2 """
        Column $col must contain ≥1 melody value and 1 human mean
        """

        push!(melodies, vals[1:end-1])   # melody
        push!(human_means, vals[end])    # final row = human mean
        push!(stemIDs, String(col))
    end

    return stemIDs, melodies, human_means
end

function load_rhythms(path::String)
    df = DataFrame(CSV.File(path, missingstring="NA"))

    rhythms = Vector{Vector{Float64}}()
    labels  = String[]

    for col in names(df)[1:end]
        vals = Float64[]

        for v in df[!, col]
            if ismissing(v)
                continue
            elseif v isa Number
                push!(vals, Float64(v))
            elseif v isa AbstractString
                parsed = tryparse(Float64, v)
                parsed === nothing || push!(vals, parsed)
            end
        end

        push!(rhythms, vals[1:end-1])
        push!(labels, String(col))
    end

    return rhythms, labels
end

function load_measures(path::String)
    df = DataFrame(CSV.File(path, missingstring="NA"))

    return [
        Int(row.length)
        for row in eachrow(df)
    ]
end

end # module DataUtils

