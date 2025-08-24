import torch

characters = '0123456789ABCDEFGHIJKLMNOPRSTUVYZ'
idx_to_char = {idx+1: char for idx, char in enumerate(characters)}
idx_to_char[0] = ''

def beam_search_ctc(outputs, beam_width=5):
    T, C = outputs.size()
    beams = [("", 0)]
    for t in range(T):
        probs = outputs[t]
        new_beams = []
        for seq, score in beams:
            for c in range(C):
                new_seq = seq
                if c != 0:
                    if len(seq) == 0 or seq[-1] != idx_to_char[c]:
                        new_seq += idx_to_char[c]
                new_score = score + torch.log(probs[c]+1e-8).item()
                new_beams.append((new_seq, new_score))
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
    return beams[0][0]
