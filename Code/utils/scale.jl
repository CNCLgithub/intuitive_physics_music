module ScaleUtils

export Note, note_values, note_positions, latent_position, latent_interval, pos0, notes
export MusicalScale, create_scale, create_major_scale, findnearest, duration, durations_from_positions
export PITCH_CLASSES, pitch_name

# 1/17/26 - Addition of Notes Struct

# A note is encoded by its value and its metrical position, P.
struct Note
    value::Float64
    position::Int64
end;

struct MusicalScale
    name::Symbol
    notes::Vector{Note}
    root::Int64
end;

# Basic pitch classes.
const PITCH_CLASSES = ["C", "C♯", "D", "E♭", "E", "F",
                       "F♯", "G", "A♭", "A", "B♭", "B"]

# Creating a scale.
function create_scale(name::Symbol, root_note::Int, root_duration::Int, intervals::Vector{Int}, measure_length::Int=8)
    notes = Note[]
    pos = 1
    for interval in intervals
        push!(notes, Note(root_note + interval, pos))
        pos = mod(pos + root_duration - 1, measure_length) + 1
    end
    
    return MusicalScale(name, notes, root_note)
end

# Creating a major scale using intervals.
function create_major_scale(root_note::Int, root_duration::Int=8)
    MAJOR_INTERVALS = [0, 2, 4, 5, 7, 9, 11, 12]
    return create_scale(:major, root_note, root_duration, MAJOR_INTERVALS)
end

# A measure is encoded by the notes inside of it and its duration
latent_position(note::Note, measure_length::Int=8) = mod(note.position - 1, measure_length) + 1
latent_interval(note1::Note, note2::Note) = note2.value - note1.value
pitch_name(v) = PITCH_CLASSES[mod(Int(round(v)), 12) + 1]

# Note values are encoded this way.
note_values(scale::MusicalScale) = [n.value for n in scale.notes]
note_positions(scale::MusicalScale) = [n.position for n in scale.notes]

function durations_from_positions(notes; L::Int=8)
    N = length(notes)
    durs = Vector{Int}(undef, N)
    
    for i in 1:N-1
        Δ = duration(notes[i], notes[i + 1])
        durs[i] = Δ > 0 ? Δ : Δ + L
    end
    durs[end] = L - notes[end].position + 1
    return durs
end

function duration(note1::Note, note2::Note, measure_length::Int=8)
    return mod(note2.position - note1.position, measure_length)
end

function findnearest(value::Note, array::Vector{Note}, exclude=nothing)
    filtered = isnothing(exclude) ? array : filter(n -> n != exclude, array)
    idx = argmin(abs.(value.value .- [n.value for n in filtered]))
    return filtered[idx]
end

# Use 0-based positions internally
pos0(note::Note) = note.position - 1  # 0..7
Δt0(note1::Note, note2::Note, measure_length::Int=8) = duration(note1, note2, measure_length)

end # module ScaleUtils

