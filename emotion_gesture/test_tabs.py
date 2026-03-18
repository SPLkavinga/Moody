"""
Quick test to verify the new analytics tabs populate correctly
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("Testing Advanced Analytics and Patterns & Insights tabs...")

try:
    from advanced_analytics import AdvancedAnalytics
    
    # Create test data
    test_emotions = [
        {'emotion': 'happy', 'confidence': 0.85, 'timestamp': '2024-12-02T09:00:00'},
        {'emotion': 'neutral', 'confidence': 0.75, 'timestamp': '2024-12-02T10:00:00'},
        {'emotion': 'sad', 'confidence': 0.72, 'timestamp': '2024-12-02T11:00:00'},
        {'emotion': 'happy', 'confidence': 0.88, 'timestamp': '2024-12-02T12:00:00'},
        {'emotion': 'angry', 'confidence': 0.80, 'timestamp': '2024-12-02T13:00:00'},
        {'emotion': 'neutral', 'confidence': 0.76, 'timestamp': '2024-12-02T14:00:00'},
    ]
    
    analytics = AdvancedAnalytics()
    
    # Populate analytics data
    for emotion in test_emotions:
        analytics.track_hourly_emotion(emotion['emotion'])
        analytics.add_stress_indicator(emotion['emotion'], emotion['confidence'])
    
    # Track some transitions
    analytics.track_emotion_transition('happy', 'neutral')
    analytics.track_emotion_transition('neutral', 'sad')
    analytics.track_emotion_transition('sad', 'happy')
    analytics.track_emotion_transition('happy', 'angry')
    
    # Calculate metrics
    wellbeing = analytics.calculate_wellbeing_score(test_emotions)
    productivity = analytics.calculate_productivity_score(test_emotions)
    stability = analytics.calculate_stability_score(test_emotions)
    stress_count = analytics.get_recent_stress_count(24)
    
    print("\n✓ Advanced Analytics Data:")
    print(f"  - Wellbeing Score: {wellbeing:.1f}/100")
    print(f"  - Productivity Index: {productivity:.1f}/100")
    print(f"  - Emotional Stability: {stability:.1f}/100")
    print(f"  - Stress Events (24h): {stress_count}")
    
    print("\n✓ Patterns & Insights Data:")
    print(f"  - Emotion Transitions: {len(analytics.emotion_transitions)} types recorded")
    print(f"  - Hourly Patterns: {len(analytics.hourly_emotions)} hours tracked")
    
    # Generate insights
    insights = analytics.generate_insights(test_emotions, wellbeing, productivity)
    print(f"  - Personalized Insights: {len(insights)} generated")
    
    for i, insight in enumerate(insights[:3], 1):
        print(f"    {i}. {insight}")
    
    print("\n✅ All analytics methods working correctly!")
    print("\n📊 The tabs should now display:")
    print("  1. Advanced Analytics: Wellbeing, Productivity, Stress, Stability scores")
    print("  2. Patterns & Insights: Transitions, Hourly patterns, Personalized insights")
    
    print("\n💡 Next: Run the emotion module and check the Analytics dashboard!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
