# Fikso wake-word detector: 10-minute talk

1. Title and one-sentence goal
2. Live demo: say "Fikso", show terminal timestamp
3. Why Polish keyword spotting is a practical problem
4. Dataset: positives, hard negatives, normal negatives
5. Honest limitation: synthetic baseline vs. real background audio
6. Pipeline diagram: waveform -> log-mel -> CNN -> probability -> streaming head
7. Why CNN over spectrograms: Chapters 22-25
8. Training details: fixed split, augmentation, runtime, checkpoint size
9. Held-out classification metrics
10. False alarms per hour on background audio
11. Error analysis with two or three concrete examples
12. What worked, what did not, and next experiment

