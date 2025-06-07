# Snippet from: timing.py
# Generated for client preview - Full code available after payment
# Contact for complete implementation

def main():
    parser = argparse.ArgumentParser(
        description="🎯 FIXED Enhanced AI Perfect Moment Detector"
    )
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument(
        "--genre",
        default="electronic",
        choices=["electronic", "hip-hop", "rock", "trap"],
        help="Music genre for optimized detection",
    )
    parser.add_argument(
        "--min_score",
        type=float,
        default=0.45,  # LOWERED default threshold
        help="Minimum ensemble score threshold",
    )
    parser.add_argument(
        "--effect_duration", type=float, default=2.0, help="Duration of each effect"
    )
    parser.add_argument(
        "--max_effects",
        type=int,
        default=5,  # Changed default to 5
        help="Maximum number of effects",
    )
    parser.add_argument("--output_json", help="Save results to JSON")
    parser.add_argument(
        "--gemini_key", help="Gemini API key (or set GEMINI_API_KEY env var)"
    )

    args = parser.parse_args()

    # Set API key if provided
    if args.gemini_key:
        os.environ["GEMINI_API_KEY"] = args.gemini_key

    try:
        print("🚀 Starting FIXED Enhanced Perfect Moment Detection...")
        detector = EnhancedPerfectMomentDetector(args.audio_file, genre=args.genre)

        # Find enhanced perfect moments with GUARANTEED count
        perfect_moments = detector.find_enhanced_perfect_moments(
            min_score=args.min_score,
            min_gap=3,  # Start with this, but will adapt
            max_effects=args.max_effects,  # Pass the target count
        )

        # Export results
        freeze_points = []
        for moment in perfect_moments[: args.max_effects]:
            start_time = float(moment["time"])
            if start_time + args.effect_duration <= detector.duration:
                freeze_points.append([start_time, args.effect_duration])
            else:
                # Adjust duration if effect would go beyond song end
                adjusted_duration = max(0.5, detector.duration - start_time)
                freeze_points.append([start_time, adjusted_duration])

        print("\n" + "=" * 80)
        print(f"🌟 ENHANCED CHROMO ZOOM MOMENTS DETECTED! (Target: {args.max_effects})")
        print("=" * 80)
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def __init__(self, audio_file: str, sr: int = 22050, genre: str = "electronic"):
        print(f"🎵 Loading audio: {audio_file}")
        self.y, self.sr = librosa.load(audio_file, sr=sr)
        self.duration = len(self.y) / self.sr
        self.genre = genre.lower()
        print(f"✅ Audio loaded: {self.duration:.2f}s, Genre: {genre}")

        # Gemini API configuration
        self.gemini_api_key = "AIzaSyCmoJhOud7q9rpgon5l5npz_aij1_6hts0"
        self.gemini_endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

        # Genre-specific parameters
        self.genre_configs = {
            "electronic": {
                "drop_detection_weight": 0.3,
                "bass_importance": 0.25,
                "buildup_sensitivity": 0.2,
                "energy_threshold": 0.4,
            },
            "hip-hop": {
                "drop_detection_weight": 0.2,
                "bass_importance": 0.35,
                "buildup_sensitivity": 0.15,
                "energy_threshold": 0.35,
                # Full code available after Monday payment confirmation
                # Contact for complete implementation


    def call_gemini_api(self, prompt: str) -> Dict:
        """
        Call Google Gemini API for LLM analysis
        """
        if not self.gemini_api_key:
            print("⚠️ GEMINI_API_KEY not found in environment variables")
            return {"error": "API key not configured"}

        headers = {"Content-Type": "application/json"}

        payload = {"contents": [{"parts": [{"text": prompt}]}]}

        try:
            url = f"{self.gemini_endpoint}?key={self.gemini_api_key}"
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()

            result = response.json()
            if "candidates" in result and result["candidates"]:
            # Full code available after Monday payment confirmation
            # Contact for complete implementation


    def extract_beat_and_rhythm_features(self):
        """Extract advanced beat, tempo, and rhythm features"""
        print("🥁 Analyzing beats and rhythm patterns...")

        # 1. Beat tracking
        tempo, beats = librosa.beat.beat_track(y=self.y, sr=self.sr, hop_length=512)
        beat_times = librosa.beat.beat_track(y=self.y, sr=self.sr, units="time")[1]

        # 2. Onset detection (sudden changes - good for drops)
        onsets = librosa.onset.onset_detect(y=self.y, sr=self.sr, units="time")
        onset_strength = librosa.onset.onset_strength(y=self.y, sr=self.sr)

        # 3. Tempo variations
        tempogram = librosa.feature.tempogram(y=self.y, sr=self.sr)
        tempo_variations = np.std(tempogram, axis=0)

        # 4. Drop detection (sudden energy increases)
        energy = librosa.feature.rms(y=self.y)[0]
        energy_smooth = gaussian_filter1d(energy, sigma=2)
        energy_diff = np.diff(energy_smooth)
        drop_candidates = find_peaks(
            energy_diff, height=np.percentile(energy_diff, 85)
            # Full code available after Monday payment confirmation
            # Contact for complete implementation


    def detect_music_structure(self):
        """Detect verse, chorus, bridge, drop sections using segment analysis"""
        print("🎵 Analyzing music structure...")

        # Chromagram for harmonic analysis
        chroma = librosa.feature.chroma_stft(y=self.y, sr=self.sr)

        # MFCC for timbral analysis
        mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=13)

        # Combine features
        features = np.vstack([chroma, mfcc])

        # Segment the song
        boundaries = librosa.segment.agglomerative(features, k=8)  # 8 segments max
        boundary_times = librosa.frames_to_time(boundaries, sr=self.sr)

        # Classify segments based on energy and harmonic content
        segments = []
        for i in range(len(boundary_times) - 1):
            start_time = boundary_times[i]
            end_time = boundary_times[i + 1]
            start_frame = int(start_time * self.sr)
            end_frame = int(end_time * self.sr)

            segment_audio = self.y[start_frame:end_frame]
            segment_energy = np.mean(librosa.feature.rms(y=segment_audio)[0])

    # Full code available after Monday payment confirmation
    # Contact for complete implementation


    def calculate_ensemble_score(
        self, features: Dict, beat_context: Dict, structure_context: Dict
    # Full code available after Monday payment confirmation


    def calculate_improved_original_score(self, features: Dict) -> float:
        """FIXED: More flexible original scoring logic"""
        score = 0.0
        max_score = 0.0

        # More flexible parameters based on your analysis
        params = {
            "vocal_score_min": 0.50,  # Lowered from 0.65
            "vocal_score_optimal": 0.65,  # Lowered from 0.718
            "pitch_stability_min": -1.0,  # More flexible
            "pitch_stability_max": 0.0,  # More flexible
            "pitch_stability_optimal": -0.5,  # More realistic
            "brightness_max": 0.35,  # More flexible
            "brightness_optimal": 0.25,  # More realistic
        }

        # Vocal score weight (more flexible)
        vocal_score = features.get("vocal_score", 0)
        if vocal_score >= params["vocal_score_min"]:
            distance = abs(vocal_score - params["vocal_score_optimal"])
            # Full code available after Monday payment confirmation
            # Contact for complete implementation


    def calculate_beat_alignment_score(self, time: float, beat_context: Dict) -> float:
        """Score based on alignment with beats"""
        beat_times = beat_context.get("beat_times", [])
        if not beat_times:
            return 0.5

        # Find closest beat
        beat_distances = [abs(time - beat_time) for beat_time in beat_times]
        min_distance = min(beat_distances)
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def calculate_drop_proximity_score(self, time: float, beat_context: Dict) -> float:
        """Score based on proximity to detected drops"""
        drop_times = beat_context.get("drop_times", [])
        if not drop_times:
            return 0.4  # Higher default score

        # Find closest drop
        drop_distances = [abs(time - drop_time) for drop_time in drop_times]
        min_distance = min(drop_distances)
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def calculate_structure_score(self, time: float, segments: List[Dict]) -> float:
        """Score based on musical structure position"""
        for segment in segments:
            if segment["start_time"] <= time <= segment["end_time"]:
            # Full code available after Monday payment confirmation
            # Contact for complete implementation


    def calculate_genre_specific_score(self, features: Dict) -> float:
        """Enhanced genre-specific scoring adjustments"""
        score = 0.5  # Base score

        if self.genre == "electronic" or self.genre == "trap":
            # Favor high energy, bass-heavy moments
            energy = features.get("energy", 0)
            bass_energy = features.get("bass_energy", 0)

            if energy > 0.35:  # Lowered threshold
                score += 0.25
            if bass_energy > 0.15:  # Lowered threshold
                score += 0.25

        elif self.genre == "hip-hop":
            # Favor rhythmic consistency and vocal presence
            # Full code available after Monday payment confirmation
            # Contact for complete implementation


    def analyze_with_gemini_context(self, top_moments: List[Dict]) -> Dict:
        """
        Use Gemini LLM to analyze patterns and provide context
        """
        try:
            # Prepare data for Gemini analysis
            moment_summary = []
            for i, moment in enumerate(top_moments[:5]):
                summary = {
                    "time": round(moment["time"], 2),
                    "ensemble_score": round(moment.get("ensemble_score", 0), 3),
                    "vocal_score": round(moment.get("vocal_score", 0), 3),
                    # Full code available after Monday payment confirmation
                    # Contact for complete implementation


    def find_enhanced_perfect_moments(
        self, min_score: float = 0.45, min_gap: float = 1.5, max_effects: int = 5
    # Full code available after Monday payment confirmation


    def extract_comprehensive_features(self, frame_duration: float = 0.5) -> List[Dict]:
        """Enhanced feature extraction with additional musical features"""
        print("🔍 Extracting comprehensive features...")

        hop_length = 512
        frame_samples = int(frame_duration * self.sr)
        overlap = frame_samples // 2

        frames = []
        for start_sample in range(0, len(self.y) - frame_samples, overlap):
            end_sample = start_sample + frame_samples
            frame_audio = self.y[start_sample:end_sample]
            frame_time = start_sample / self.sr
            # Full code available after Monday payment confirmation
            # Contact for complete implementation


    def analyze_enhanced_frame(self, frame_audio: np.ndarray, time: float) -> Dict:
        """Enhanced frame analysis with additional features"""
        features = {"time": float(time)}

        # Original features (simplified)
        y_harmonic, y_percussive = librosa.effects.hpss(frame_audio)
        harmonic_energy = np.sum(y_harmonic**2)
        total_energy = np.sum(frame_audio**2)
        features["vocal_score"] = float(harmonic_energy / (total_energy + 1e-8))

        # Pitch analysis
        pitches, magnitudes = librosa.piptrack(y=frame_audio, sr=self.sr, threshold=0.1)
        pitch_candidates = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 80:
                pitch_candidates.append(float(pitch))

        if pitch_candidates:
            features["pitch"] = float(np.mean(pitch_candidates))
            features["pitch_stability"] = float(
                -np.std(pitch_candidates) / (np.mean(pitch_candidates) + 1e-8)
            )
        else:
            features["pitch"] = 0.0
            features["pitch_stability"] = -1.0

        # Enhanced features
        stft = librosa.stft(frame_audio)
        # Full code available after Monday payment confirmation
        # Contact for complete implementation

