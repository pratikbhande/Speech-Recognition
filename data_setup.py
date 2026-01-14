"""Setup sample speakers with BOTH SpeechBrain and WeSpeaker embeddings."""

from pathlib import Path
import soundfile as sf
import config
from voice_embeddings import VoiceEmbedder
from database import MongoManager, QdrantManager
from datetime import datetime
from audio_preprocessing import preprocess_audio
import numpy as np


def setup_samples():
    """Enroll all sample speakers with BOTH models for comparison."""
    
    # Initialize BOTH models
    print("\n" + "=" * 60)
    print("🤖 Initializing BOTH models for comparison...")
    print("=" * 60)
    
    try:
        embeddings_speechbrain = VoiceEmbedder(model_type="speechbrain")
    except Exception as e:
        print(f"❌ Failed to load SpeechBrain: {e}")
        embeddings_speechbrain = None
    
    try:
        embeddings_wespeaker = VoiceEmbedder(model_type="wespeaker")
    except Exception as e:
        print(f"❌ Failed to load WeSpeaker: {e}")
        embeddings_wespeaker = None
    
    if not embeddings_speechbrain and not embeddings_wespeaker:
        print("❌ No models available!")
        return
    
    mongo = MongoManager()
    qdrant = QdrantManager()
    
    samples_dir = Path("data/samples")
    if not samples_dir.exists():
        print(f"❌ Directory not found: {samples_dir}")
        return
    
    audio_files = list(samples_dir.glob("*"))
    audio_files = [f for f in audio_files if f.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']]
    
    if not audio_files:
        print("❌ No audio files found!")
        return
    
    audio_files.sort(key=lambda x: x.name)
    
    print(f"\n🔍 Found {len(audio_files)} audio files")
    print("📊 Will enroll with BOTH models for comparison")
    print("=" * 60)
    
    enrolled_count = 0
    
    for idx, audio_file in enumerate(audio_files[:10], 1):
        try:
            print(f"\n[{idx}/10] Processing: {audio_file.name}")
            
            # Load audio
            audio, sr = sf.read(audio_file)
            print(f"  ✓ Loaded: {len(audio)/sr:.1f}s @ {sr}Hz")
            
            # Convert to mono
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # Preprocess
            audio_processed = preprocess_audio(audio, sr, for_enrollment=True)
            print(f"  ✓ Preprocessed: {len(audio_processed)/sr:.1f}s")
            
            # Safety check
            if len(audio_processed) < sr * 1.0:
                print(f"  ⚠️ Using original audio")
                audio_processed = audio
                max_val = np.abs(audio_processed).max()
                if max_val > 0:
                    audio_processed = audio_processed / max_val
            
            # Validate
            if len(audio_processed) < sr * 1.0:
                print(f"  ⚠️ Skipped: Too short")
                continue
            
            # Truncate if needed
            if len(audio_processed) > sr * 30:
                audio_processed = audio_processed[:sr * 30]
                print(f"  ✓ Truncated to 30s")
            
            speaker_name = config.SAMPLE_SPEAKERS[idx - 1] if idx <= len(config.SAMPLE_SPEAKERS) else f"Speaker {idx}"
            
            # ========== ENROLL WITH SPEECHBRAIN ==========
            if embeddings_speechbrain:
                try:
                    print(f"  🔵 Extracting SpeechBrain embedding...")
                    client_id_sb = f"SAMPLE_{idx:02d}_SB_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
                    
                    embedding_sb = embeddings_speechbrain.extract_from_array(audio_processed, sr)
                    print(f"     ✓ SpeechBrain embedding: {embedding_sb.shape}")
                    
                    # Store in MongoDB
                    user_doc_sb = {
                        "client_id": client_id_sb,
                        "name": f"{speaker_name}",
                        "created_at": datetime.utcnow(),
                        "is_sample": True,
                        "model_type": "speechbrain"
                    }
                    mongo.users.insert_one(user_doc_sb)
                    
                    # Store in Qdrant
                    point_id_sb = hash(client_id_sb) & 0x7FFFFFFF
                    qdrant.insert(point_id_sb, embedding_sb, {"client_id": client_id_sb})
                    
                    print(f"     ✅ SpeechBrain enrolled: {client_id_sb}")
                except Exception as e:
                    print(f"     ❌ SpeechBrain failed: {str(e)}")
            
            # ========== ENROLL WITH WESPEAKER ==========
            if embeddings_wespeaker:
                try:
                    print(f"  🟢 Extracting WeSpeaker embedding...")
                    client_id_ws = f"SAMPLE_{idx:02d}_WS_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
                    
                    embedding_ws = embeddings_wespeaker.extract_from_array(audio_processed, sr)
                    print(f"     ✓ WeSpeaker embedding: {embedding_ws.shape}")
                    
                    # Store in MongoDB
                    user_doc_ws = {
                        "client_id": client_id_ws,
                        "name": f"{speaker_name}",
                        "created_at": datetime.utcnow(),
                        "is_sample": True,
                        "model_type": "wespeaker"
                    }
                    mongo.users.insert_one(user_doc_ws)
                    
                    # Store in Qdrant
                    point_id_ws = hash(client_id_ws) & 0x7FFFFFFF
                    qdrant.insert(point_id_ws, embedding_ws, {"client_id": client_id_ws})
                    
                    print(f"     ✅ WeSpeaker enrolled: {client_id_ws}")
                except Exception as e:
                    print(f"     ❌ WeSpeaker failed: {str(e)}")
            
            # Save processed audio
            new_name = samples_dir / f"sample_{idx:02d}.wav"
            sf.write(new_name, audio_processed, sr)
            
            if audio_file != new_name and audio_file.exists():
                audio_file.unlink()
            
            print(f"  ✅ Enrolled: {speaker_name} with BOTH models")
            enrolled_count += 1
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 60)
    print(f"✅ Enrollment complete! {enrolled_count}/{min(len(audio_files), 10)} samples")
    print(f"📊 Each sample enrolled with BOTH models:")
    if embeddings_speechbrain:
        print(f"   🔵 SpeechBrain: {enrolled_count} samples")
    if embeddings_wespeaker:
        print(f"   🟢 WeSpeaker: {enrolled_count} samples")
    print(f"\nTotal enrollments in database: {enrolled_count * 2}")
    print("\n🎯 Now you can compare both models in the app!")
    print("   - Switch model in sidebar")
    print("   - Test samples to compare accuracy")
    print("\nRun: streamlit run app.py")


if __name__ == "__main__":
    setup_samples()