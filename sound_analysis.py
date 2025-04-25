import os
import librosa
import soundfile as sf
import numpy as np
import joblib
import webrtcvad
import wave
# from moviepy.editor import VideoFileClip
import moviepy

def is_voice_present(audio_path, threshold=0.1):
    vad = webrtcvad.Vad(0)  # 0: 민감함, 3: 덜 민감
    with wave.open(audio_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        if sample_rate != 16000 or wf.getnchannels() != 1:
            return False  # 사전 처리 필요

        frame_duration = 30  # ms
        frame_size = int(sample_rate * frame_duration / 1000) * 2
        audio = wf.readframes(wf.getnframes())

        speech_frames = 0
        total_frames = 0

        for i in range(0, len(audio), frame_size):
            frame = audio[i:i + frame_size]
            if len(frame) < frame_size:
                continue
            total_frames += 1
            if vad.is_speech(frame, sample_rate):
                speech_frames += 1

        if total_frames == 0:
            return False

        speech_ratio = speech_frames / total_frames
        return speech_ratio > threshold

def extract_librosa_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    features = {}

    # 1. Tempo (BPM)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = tempo

    # 2. Chroma STFT (12-dim)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i, val in enumerate(np.mean(chroma, axis=1)):
        features[f'chroma_stft_{i}'] = val

    # 3. MFCC (Mel Frequency Cepstral Coefficients) - 20개 기본
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i, val in enumerate(np.mean(mfcc, axis=1)):
        features[f'mfcc_{i}'] = val

    # 4. Spectral Centroid
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid'] = np.mean(spec_centroid)

    # 5. Spectral Bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['spectral_bandwidth'] = np.mean(spec_bw)

    # 6. Spectral Rolloff
    spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['spectral_rolloff'] = np.mean(spec_rolloff)

    # 7. Spectral Contrast (7 bands)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i, val in enumerate(np.mean(spec_contrast, axis=1)):
        features[f'spectral_contrast_{i}'] = val

    # 8. Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zero_crossing_rate'] = np.mean(zcr)

    # 9. RMS Energy
    rms = librosa.feature.rms(y=y)
    features['rms_energy'] = np.mean(rms)

    # 10. Tonnetz (6-dim tonality)
    y_harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    for i, val in enumerate(np.mean(tonnetz, axis=1)):
        features[f'tonnetz_{i}'] = val

    # 11. Onset Strength (리듬 변화 강도)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    features['onset_strength_mean'] = np.mean(onset_env)
    features['onset_strength_std'] = np.std(onset_env)

    # 12. Harmonic & Percussive energy ratio
    y_harm, y_perc = librosa.effects.hpss(y)
    features['harmonic_energy'] = np.mean(np.abs(y_harm))
    features['percussive_energy'] = np.mean(np.abs(y_perc))
    features['harmonic_percussive_ratio'] = features['harmonic_energy'] / (features['percussive_energy'] + 1e-6)

    # 13. Duration
    features['duration'] = librosa.get_duration(y=y, sr=sr)

    return features


def analyze_music_mood_prob(audio_path):
    y, sr = librosa.load(audio_path)

    # 주요 feature 추출
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr).mean()
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()

    # 정규화
    tempo_score = min(tempo / 200, 1.0)  # 0~1
    chroma_score = min(chroma_stft / 1.0, 1.0)  # 대략 0~1
    centroid_score = min(spectral_centroid / 4000, 1.0)

    # 가중 평균 방식으로 soft score 계산
    happy_score = 0.6 * tempo_score + 0.4 * chroma_score
    calm_score = 1.0 - centroid_score  # 낮은 centroid = 부드러운 소리
    neutral_score = 1.0 - abs(happy_score - calm_score)

    # 정규화
    total = happy_score + calm_score + neutral_score + 1e-6
    scores = {
        "happy/energetic": happy_score / total,
        "calm/sad": calm_score / total,
        "neutral": neutral_score / total
    }

    # 가장 높은 점수의 감정 선택
    mood = max(scores, key=scores.get)
    return mood, scores


from speechbrain.pretrained import EncoderClassifier

# 모델 로드
emotion_model = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir="pretrained_models/emotion-recognition"
)

from transformers import pipeline

# Load only once
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-base")
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_voice_sentiment(audio_path):
    print("🗣️ Using Whisper + Sentiment pipeline...")
    try:
        transcription = asr_pipeline(audio_path)
        text = transcription["text"]
        print(f"📝 Transcribed Text: {text}")

        sentiment_result = sentiment_pipeline(text)[0]
        print(f"🔍 Sentiment: {sentiment_result['label']}, Score: {sentiment_result['score']:.2f}")
        return sentiment_result['label'].lower(), sentiment_result['score']
    except Exception as e:
        print(f"❌ Error during Whisper/Sentiment analysis: {e}")
        return "error", 0.0


def convert_to_mono_wav(input_path, output_path, target_sr=16000):
    y, sr = librosa.load(input_path, sr=target_sr, mono=True)
    sf.write(output_path, y, target_sr)

def continuous_voice_sentiment_score(label, score):
    if label == "positive":
        return score  # e.g., 0.98 → +0.98
    elif label == "negative":
        return -score  # e.g., 0.87 → -0.87
    else:
        return 0.0


def extract_audio_sentiment(video_path):
    original_audio_path = "temp_audio_original.wav"
    converted_audio_path = "temp_audio.wav"

    clip = moviepy.editor.VideoFileClip(video_path)
    clip.audio.write_audiofile(original_audio_path, verbose=False, logger=None)

    convert_to_mono_wav(original_audio_path, converted_audio_path)

    sentiment_voice = None
    score_voice = None
    sentiment_music = None
    score_music = {}

    if is_voice_present(converted_audio_path):
        print("🎤 Detected human voice. Running both speech and music sentiment...")
        sentiment_voice, score_voice = analyze_voice_sentiment(converted_audio_path)
        sentiment_music, score_music = analyze_music_mood_prob(converted_audio_path)
        final_sentiment = sentiment_voice if sentiment_voice != "neutral" else sentiment_music
    else:
        print("🎵 No voice detected. Using music mood...")
        sentiment_music, score_music = analyze_music_mood_prob(converted_audio_path)
        final_sentiment = sentiment_music

    # librosa feature 추출
    audio_features = extract_librosa_features(converted_audio_path)

    # 연속 점수화
    voice_sentiment_score = continuous_voice_sentiment_score(sentiment_voice, score_voice)
    music_sentiment_continuous = score_music["happy/energetic"] - score_music["calm/sad"]

    result = {
        "sentiment_voice": sentiment_voice,
        "score_voice": score_voice,
        "voice_sentiment_score": voice_sentiment_score,
        "sentiment_music": sentiment_music,
        "score_music_happy_energetic": score_music.get("happy/energetic", 0),
        "score_music_calm_sad": score_music.get("calm/sad", 0),
        "score_music_neutral": score_music.get("neutral", 0),
        "music_sentiment_score": music_sentiment_continuous,
        "final_sentiment": final_sentiment
    }

    # 결과에 librosa 특성 추가
    result.update(audio_features)

    return result




def process_video(path, filename):
    print(f"Processing video: {filename}")
    full_path = os.path.join(path, filename)
    audio_sentiment = extract_audio_sentiment(full_path)
    return audio_sentiment

import pandas as pd
import os
import numpy as np

# 결과 저장 리스트
data = []

# 분석할 디렉토리 경로
path = '/Users/manyyeon/data/UNM/RA/instagram/video_data'

# 결과에 포함할 메타데이터 필드 (필요 없으면 제거 가능)
METADATA_FIELDS = ['filename']  # 예시: filename만 사용
FEATURE_COLUMNS = []  # 자동으로 채울 거라 비워둠

for filename in os.listdir(path):
    name, ext = os.path.splitext(filename)

    if ext.lower() not in [".mp4"]:
        continue

    print(f"▶️ Processing: {filename}")
    result_dict = process_video(path, filename)

    # 결과 딕셔너리에 filename 추가
    result_dict['filename'] = filename

    # 첫 번째 파일에서 컬럼명 저장
    if not FEATURE_COLUMNS:
        FEATURE_COLUMNS = list(result_dict.keys())

    # 결과를 순서대로 리스트로 변환해서 저장
    row = [result_dict.get(col, np.nan) for col in FEATURE_COLUMNS]
    data.append(row)

# DataFrame 생성
df = pd.DataFrame(data, columns=FEATURE_COLUMNS)

# CSV로 저장
output_path = 'audio_sentiment_analysis_results.csv'
df.to_csv(output_path, index=False)

print(f"✅ Processing complete! CSV saved to: {os.path.abspath(output_path)}")