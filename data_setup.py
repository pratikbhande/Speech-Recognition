"""Setup sample speakers with audio preprocessing."""

from pathlib import Path
import soundfile as sf
import config
from voice_embeddings import VoiceEmbedder  # VoiceEmbedder class
from database import MongoManager, QdrantManager
from datetime import datetime
from audio_preprocessing import preprocess_audio


def setup_samples():
    """Enroll all sample speakers with preprocessing."""
    
    embeddings = VoiceEmbedder()  # VoiceEmbedder instance
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
    
    print(f"\n📁 Found {len(audio_files)} audio files")
    print("=" * 60)
    
    enrolled_count = 0
    
    for idx, audio_file in enumerate(audio_files[:10], 1):  # Max 10 samples
        try:
            print(f"\n[{idx}/10] Processing: {audio_file.name}")
            
            # Load audio
            audio, sr = sf.read(audio_file)
            print(f"  ✓ Loaded: {len(audio)/sr:.1f}s @ {sr}Hz")
            
            # PREPROCESSING: Clean audio before enrollment
            audio = preprocess_audio(audio, sr, enhance=True)
            print(f"  ✓ Preprocessed: {len(audio)/sr:.1f}s (cleaned)")
            
            # Validate
            if len(audio) < sr * 1:
                print(f"  ⚠️ Skipped: Too short after VAD")
                continue
            
            if len(audio) > sr * 30:
                audio = audio[:sr * 30]
                print(f"  ✓ Truncated to 30s")
            
            # Get speaker name
            speaker_name = config.SAMPLE_SPEAKERS[idx - 1]
            
            # Generate client ID
            client_id = f"SAMPLE_{idx:02d}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            
            # Extract embedding - FIXED: use extract_from_array()
            embedding = embeddings.extract_from_array(audio, sr)
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
            
            # Rename file
            new_name = samples_dir / f"sample_{idx:02d}.wav"
            sf.write(new_name, audio, sr)
            if audio_file != new_name:
                audio_file.unlink()
            
            print(f"  ✅ Enrolled: {speaker_name} ({client_id})")
            enrolled_count += 1
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            continue
    
    print("\n" + "=" * 60)
    print(f"✅ Enrollment complete! {enrolled_count}/10 samples enrolled")
    print("\nRun: streamlit run app.py")


if __name__ == "__main__":
    setup_samples()