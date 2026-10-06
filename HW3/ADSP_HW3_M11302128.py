# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 00:20:34 2025

@author: Jessica
"""

import numpy as np
from scipy.io.wavfile import write

# === 內建簡譜資料 ===
score = ["5'", "3'", "4'", "5'", "3'", "4'", 
         "5'", "5", "6", "7", "1'", "2'", "3'", "4'",
         "3'", "1'", "2'", "3'", "3", "4", 
         "5", "6", "5", "4", "5", "3", "4", "5",
         "4", "6", "5", "4", "3", "2", 
         "3", "2", "1", "2", "3", "4", "5", "6",
         "4", "6", "5", "6", "7", "1'", 
         "5", "6", "7", "1'", "2'", "3'", "4'", "5'",
         "5'", "5'", "5'", "6'", "5'", "4'",
         "3'", "3'", "3'", "4'", "3'", "2'", "1'",
         "1'", "2'", "1'", "7", "6", "1'","0"]

beat = [1, 0.5, 0.5, 1, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        1, 0.5, 0.5, 1, 0.5, 0.5, 
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        1, 0.5, 0.5, 1, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        1, 0.5, 0.5, 1, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        2.5, 1, 1, 1, 1, 1, 
        2.5, 1, 1, 1, 1, 1, 2.5, 
        1, 1, 1, 1, 2.5, 1, 3]

name = 'music'

# === 開始產生音樂 ===
fs = 44100                                              # 取樣頻率
duration_unit = 0.4                                     # 一拍是0.4秒 
sound_ratio = 0.95                                      # 聲音佔95%

# 音高 (C大調)
notes_freq = {
    '1': 261.63, '2': 293.66, '3': 329.63, '4': 349.23, '5': 392.00, 
    '6': 440.00, '7': 493.88,
    "1'": 523.25, "2'": 587.33, "3'": 659.25, "4'": 698.46, "5'": 783.99, 
    "6'": 880.00, "7'": 987.77,
    '0': 0
}

def generate_song(score, beat, transpose_ratio=1.0):
    song = np.array([], dtype=np.float32)
    previous_note = None 
    
    for note, b in zip(score, beat):
        f = notes_freq.get(note, 0)
        total_duration = b * duration_unit
        sound_duration = total_duration * sound_ratio
        rest_duration = total_duration - sound_duration 
    
        t = np.linspace(0, sound_duration, int(fs * sound_duration), endpoint=False)
        
        volume = 1.0
        if note == previous_note:                               # 1.連續音 第二個振幅增加
            volume = 2
    
        if f == 0:
            waveform = np.zeros_like(t)
        else:
            fundamental = np.sin(2 * np.pi * f * t)             # 基本音
            harmonic2 = 0.3 * np.sin(2 * np.pi * 2 * f * t)     # 2倍頻
            harmonic3 = 0.15 * np.sin(2 * np.pi * 3 * f * t)    # 3倍頻
            harmonic4 = 0.05 * np.sin(2 * np.pi * 4 * f * t)    # 4倍頻
            
            waveform = fundamental + harmonic2 + harmonic3 + harmonic4
    
           
            attack_time = 0.2                                   # 2. 開頭音量漸強
            release_time = 0.4                                  #    結尾音量漸弱
            attack_samples = int(len(t) * attack_time)
            release_samples = int(len(t) * release_time)
            envelope = np.ones_like(t) 
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
            envelope[-release_samples:] = np.linspace(1, 0, release_samples)
    
            waveform = waveform * envelope * volume
                                                              
        rest = np.zeros(int(fs * rest_duration))                # 斷音處理：每個音後面接一小段靜音
        song = np.concatenate((song, waveform, rest))           # 3.倍頻  4.音量調整  5.斷音
        previous_note = note
    
    return song

song1 = generate_song(score, beat)

transpose_semitones = 2
transpose_ratio = 2 ** (transpose_semitones / 12)               # 6.升高兩個半音 重播換D大調
song2 = generate_song(score, beat, transpose_ratio=transpose_ratio)

final_song = np.concatenate((song1, song2))                     # 兩段合併


final_song = final_song * (32767 / np.max(np.abs(final_song)))
final_song = final_song.astype(np.int16)
write(f"{name}.wav", fs, final_song)

print("Successfully")
