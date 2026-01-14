"""Setup sample speakers with minimal preprocessing."""

from pathlib import Path
import soundfile as sf
import config
from voice_embeddings import VoiceEmbedder
from database import MongoManager, QdrantManager
from datetime import datetime
from audio_preprocessing import preprocess_audio
import numpy as np


def setup_samples():
    """Enroll all sample speakers with MINIMAL preprocessing."""
    
    embeddings = VoiceEmbedder()
    mongo = MongoManager()
    qdrant = QdrantManager()
    
    samples_dir = Path("data/samples")
    if not samples_dir.exists():
        print(f"❌ Directory not found: {samples_dir}")
        return
    
    # Get all audio files
    audio_files = list(samples_dir.glob("*"))
    audio_files = [f for f in audio_files if f.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']]
    
    if not audio_files:
        print("❌ No audio files found!")
        return
    
    # CRITICAL FIX: Sort files by name to ensure consistent order
    audio_files.sort(key=lambda x: x.name)
    
    print(f"\n🔍 Found {len(audio_files)} audio files")
    print("=" * 60)
    
    enrolled_count = 0
    
    for idx, audio_file in enumerate(audio_files[:10], 1):
        try:
            print(f"\n[{idx}/10] Processing: {audio_file.name}")
            
            # Load audio
            audio, sr = sf.read(audio_file)
            print(f"  ✓ Loaded: {len(audio)/sr:.1f}s @ {sr}Hz")
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            
            # MINIMAL preprocessing for enrollment - preserve voice characteristics
            audio_processed = preprocess_audio(audio, sr, for_enrollment=True)
            print(f"  ✓ Preprocessed: {len(audio_processed)/sr:.1f}s")
            
            # Safety check - if preprocessing removed too much, use original
            if len(audio_processed) < sr * 1.0:
                print(f"  ⚠️ Preprocessing too aggressive, using original audio")
                audio_processed = audio
                # Just normalize
                max_val = np.abs(audio_processed).max()
                if max_val > 0:
                    audio_processed = audio_processed / max_val
            
            # Validate minimum length
            if len(audio_processed) < sr * 1.0:
                print(f"  ⚠️ Skipped: Too short (need >1s, got {len(audio_processed)/sr:.1f}s)")
                continue
            
            # Truncate if too long (keep first 30s)
            if len(audio_processed) > sr * 30:
                audio_processed = audio_processed[:sr * 30]
                print(f"  ✓ Truncated to 30s")
            
            # Get speaker name
            if idx <= len(config.SAMPLE_SPEAKERS):
                speaker_name = config.SAMPLE_SPEAKERS[idx - 1]
            else:
                speaker_name = f"Speaker {idx}"
            
            # Generate client ID
            client_id = f"SAMPLE_{idx:02d}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            
            # Extract embedding
            embedding = embeddings.extract_from_array(audio_processed, sr)
            print(f"  ✓ Embedding: {embedding.shape}")
            
            # Store in MongoDB
            user_doc = {
                "client_id": client_id,
                "name": speaker_name,
                "created_at": datetime.utcnow(),
                "is_sample": True
            }
            mongo.users.insert_one(user_doc)
            
            # Store in Qdrant
            point_id = hash(client_id) & 0x7FFFFFFF
            qdrant.insert(point_id, embedding, {"client_id": client_id})
            
            # Save processed file with correct name
            new_name = samples_dir / f"sample_{idx:02d}.wav"
            sf.write(new_name, audio_processed, sr)
            
            # Delete old file only if it has a different name
            if audio_file != new_name and audio_file.exists():
                audio_file.unlink()
            
            print(f"  ✅ Enrolled: {speaker_name} ({client_id})")
            enrolled_count += 1
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 60)
    print(f"✅ Enrollment complete! {enrolled_count}/{min(len(audio_files), 10)} samples enrolled")
    print("\nRun: streamlit run app.py")


if __name__ == "__main__":
    setup_samples()