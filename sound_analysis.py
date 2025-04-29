import os
import librosa
import soundfile as sf
import numpy as np
import joblib
import webrtcvad
import wave
# import moviepy.editor as mp
import ffmpeg

def extract_audio_from_video(video_path, audio_path):
    # 먼저 오디오 스트림이 있는지 체크
    probe = ffmpeg.probe(video_path)
    audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']

    if not audio_streams:
        raise ValueError(f"No audio stream found in {video_path}")

    # 오디오 스트림 추출
    stream = ffmpeg.input(video_path)
    stream = ffmpeg.output(stream.audio, audio_path)
    ffmpeg.run(stream, overwrite_output=True, quiet=True)

    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        raise ValueError(f"❌ Audio file not created or empty: {audio_path}")

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


from speechbrain.inference import EncoderClassifier

# 모델 로드
emotion_model = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir="pretrained_models/emotion-recognition"
)

from transformers import pipeline

# Load only once
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-base", return_timestamps=True)
sentiment_pipeline = pipeline("sentiment-analysis")

def analyze_voice_sentiment(audio_path):
    print("🗣️ Using Whisper + Sentiment pipeline...")
    try:
        transcription = asr_pipeline(audio_path)
        print('transcription: ', transcription)
        if "chunks" in transcription:
            text = " ".join(chunk["text"] for chunk in transcription["chunks"])
        else:
            text = transcription["text"]

        print(f"📝 Transcribed Text: {text}")

        if len(text.strip()) < 3:
            print("⚠️ Text too short for sentiment analysis.")
            return "neutral", 0.0, text

        sentiment_result = sentiment_pipeline(text)[0]
        print(f"🔍 Sentiment: {sentiment_result['label']}, Score: {sentiment_result['score']:.2f}")
        return sentiment_result['label'].lower(), sentiment_result['score'], text
    except Exception as e:
        print(f"❌ Error during Whisper/Sentiment analysis: {e}")
        return "error", 0.0, ""


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

    try:
        extract_audio_from_video(video_path, original_audio_path)
    except ValueError as e:
        print(f"⚠️ Skipping {video_path}: {e}")
        return None  # 분석할 오디오가 없음

    convert_to_mono_wav(original_audio_path, converted_audio_path)

    sentiment_voice = None
    score_voice = None
    extracted_text_from_voice = None
    sentiment_music = None
    score_music = {}

    if is_voice_present(converted_audio_path):
        print("🎤 Detected human voice. Running both speech and music sentiment...")
        sentiment_voice, score_voice, extracted_text_from_voice = analyze_voice_sentiment(converted_audio_path)
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
        "extracted_text_from_voice": extracted_text_from_voice,
        "sentiment_music": sentiment_music,
        "score_music_happy_energetic": score_music.get("happy/energetic", 0),
        "score_music_calm_sad": score_music.get("calm/sad", 0),
        "score_music_neutral": score_music.get("neutral", 0),
        "music_sentiment_score": music_sentiment_continuous,
        "final_sentiment": final_sentiment,
    }

    # 결과에 librosa 특성 추가
    result.update(audio_features)

    return result




def process_video(path, filename):
    print(f"Processing video: {filename}")
    full_path = os.path.join(path, filename)
    audio_sentiment = extract_audio_sentiment(full_path)
    return audio_sentiment


def analyze_audio_from_video_directory(path: str, output_csv: str = "audio_sentiment_analysis_results.csv"):
    import os
    import pandas as pd
    import numpy as np

    data = []
    FEATURE_COLUMNS = []

    for filename in os.listdir(path):
        name, ext = os.path.splitext(filename)
        if ext.lower() != ".mp4":
            continue

        print(f"▶️ Processing: {filename}")
        result_dict = process_video(path, filename)
        if result_dict is None:
            print(f"⏩ Skipped {filename} because no audio was found.")
            continue

        result_dict['filename'] = filename

        if not FEATURE_COLUMNS:
            FEATURE_COLUMNS = list(result_dict.keys())

        row = [result_dict.get(col, np.nan) for col in FEATURE_COLUMNS]
        data.append(row)

        # Save CSV after each file
        df = pd.DataFrame(data, columns=FEATURE_COLUMNS)
        df.to_csv(output_csv, index=False)

    print(f"✅ Processing complete! CSV saved to: {os.path.abspath(output_csv)}")
    return df

def main():
    print("Starting audio sentiment analysis...")
    # df_result = analyze_audio_from_video_directory(
    #     path='/Users/dayeon/data/UNM/RA/instagram_project/burberry',
    #     output_csv = 'results_burberry.csv'
    # )
    # analyze_audio_from_video_directory(
    #     path='/Users/dayeon/data/UNM/RA/instagram_project/cartier',
    #     output_csv = 'results_cartier.csv'
    # )
    analyze_audio_from_video_directory(
        path='/Users/dayeon/data/UNM/RA/instagram_project/test',
        output_csv = 'results_test.csv'
    )
    analyze_audio_from_video_directory(
        path='/Users/dayeon/data/UNM/RA/instagram_project/chanelofficial',
        output_csv = 'results_chanelofficial.csv'
    )
    analyze_audio_from_video_directory(
        path='/Users/dayeon/data/UNM/RA/instagram_project/dior',
        output_csv = 'results_dior.csv'
    )
    analyze_audio_from_video_directory(
        path='/Users/dayeon/data/UNM/RA/instagram_project/gucci',
        output_csv = 'results_gucci.csv'
    )
    analyze_audio_from_video_directory(
        path='/Users/dayeon/data/UNM/RA/instagram_project/hermes',
        output_csv = 'results_hermes.csv'
    )
    analyze_audio_from_video_directory(
        path='/Users/dayeon/data/UNM/RA/instagram_project/louisvuitton',
        output_csv = 'results_louisvuitton.csv'
    )
    analyze_audio_from_video_directory(
        path='/Users/dayeon/data/UNM/RA/instagram_project/prada',
        output_csv = 'results_prada.csv'
    )
    analyze_audio_from_video_directory(
        path='/Users/dayeon/data/UNM/RA/instagram_project/tiffanyandco',
        output_csv = 'results_tiffanyandco.csv'
    )

if __name__ == "__main__":
    main()