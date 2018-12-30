from mido import Message, MidiFile, MidiTrack, second2tick


#dims for one-hot representation
note_dim = 128
velocity_dim = 16
duration_dim = 161
dtime_dim = 161


#TIME TRANSFORMATIONS
def real_to_discrete(value,max_sec,scale):
    return round(min(value/max_sec,1)*scale)
def discrete_to_real(value,max_sec,scale):
    return max_sec*value/scale
def discrete_to_ticks(value, max_sec, scale,ticks_per_beat,tempo):
    return int(round(ticks_per_beat*max_sec/tempo/scale*value))


def midi_to_input_seq(mid):
    sequence = []
    current_notes = {}
    time = 0
    for msg in mid:
        time += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            current_notes.update({(msg.note, msg.channel): [msg.velocity,time]})
        if msg.type == 'note_off' or ( msg.type == 'note_on' and msg.velocity == 0):
            v = current_notes.pop((msg.note, msg.channel), None)
            if v == None: continue
            note_velocity, note_time = v
            sequence.append([msg.note,note_velocity/127,time-note_time,note_time])
    sequence.sort(key = lambda x: x[-1])
    seq = [sequence[0][:-1]+[0.0]]
    for i in range(1,len(sequence)):
        seq.append(sequence[i][:-1])
        seq[i].append(sequence[i][-1]-sequence[i-1][-1])
    return seq

# vector in input_seq =  [note(int),velocity(float),duration(float),dtime(float)]

def input_seq_to_output_seq(input_seq):
    output_seq = []
    for v in input_seq:
        v_out = [v[0], round(v[1]*15), real_to_discrete(v[2],4,duration_dim-1), real_to_discrete(v[3],4,dtime_dim-1)]
        output_seq.append(v_out)
    return output_seq


def output_seq_to_input_seq(output_seq):
    input_seq = []
    for v in output_seq:
        v_out = [v[0], v[1]/15, discrete_to_real(v[2],4,duration_dim-1), discrete_to_real(v[3],4,dtime_dim-1)]
        input_seq.append(v_out)
    return input_seq


def output_seq_to_midi(seq):
    tempo = 0.5 # (500000)  mido: when create new track you need time in ticks
    ticks_per_beat = 480 # time_in_seconds = tempo * 1e-6 * time_in_ticks / ticks_per_beat
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(Message('program_change', program=0, time=0))
    notes = []
    time = 0
    for v in seq:
        note,velocity = v[0], round(127/15*v[1])
        duration = discrete_to_ticks(v[2],4,duration_dim-1,ticks_per_beat,tempo)
        dtime = discrete_to_ticks(v[3],4,dtime_dim-1,ticks_per_beat,tempo)
        time += dtime
        notes.append(['note_on',int(note),int(velocity),time])
        notes.append(['note_off',int(note),64,time+duration])
    notes.sort(key=lambda x: x[-1])
    track.append(Message(notes[0][0], note=notes[0][1], velocity=notes[0][2], time=0))
    for i in range(1,len(notes)):
        track.append( Message(notes[i][0],  note=notes[i][1], velocity=notes[i][2], time=notes[i][3]-notes[i-1][3]) )
    return mid
