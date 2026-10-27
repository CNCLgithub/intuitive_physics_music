module ForceUtils
using ..ScaleUtils
export calculate_gravity, calculate_magnetism, calculate_inertia

## Gravity pulls a note toward the nearest stable scale degree below it, scaling with distance (Larson, 1993, 2004)
function calculate_gravity(note::Note, key::MusicalScale)
    
    below = filter(n -> n.value < note.value, key.notes)
    
    isempty(below) && return 0.0

    nearest_below = below[argmax([n.value for n in below])]  # closest one from below
    
    return -1.0 * abs(note.value - nearest_below.value)
end
;

## Finds the difference between the inverse squares of the distance to the nearest stable pitch in both directions. Adapted from Larson, 2004, p. 458.
function magnetism_potential(note::Note, key::MusicalScale)
    # Finds 2 nearest stable pitches to current note
    nearest = findnearest(note, key.notes)
    second_nearest = findnearest(note, key.notes, nearest)

    # Finds distance between the note and each pitch
    dto   = abs(note.value - nearest.value)
    dfrom = abs(note.value - second_nearest.value)

    # Takes difference of inverse squares of dto and dfrom
    magnetism_force = (1 / (dto^2 + 1)) - (1 / (dfrom^2 + 1))

    return magnetism_force
end

# Incorporation of directional magnetism gradient
function calculate_magnetism(note::Note, key::MusicalScale; ϵ = 1e-3)
    U0 = magnetism_potential(note, key)

    note_up   = Note(note.value + ϵ, note.position)
    note_down = Note(note.value - ϵ, note.position)

    U_up   = magnetism_potential(note_up, key)
    U_down = magnetism_potential(note_down, key)

    return -(U_up - U_down) / (2ϵ)
end
;

function calculate_inertia(prev_note::Note, curr_note::Note, key::MusicalScale)
    Δ = curr_note.value - prev_note.value

    scale_values = [n.value for n in key.notes]
    degree = findfirst(v -> v == curr_note.value, scale_values)

    base = 0.25
    deg_factor = isnothing(degree) ? 1.0 : 0.6 + 0.05 * degree
    strength = min(0.45, base * deg_factor)

    if Δ > 0
        return +strength
    elseif Δ < 0
        return -strength
    else
        return 0.0
    end
end
;

end # module ForceUtils
