"""
Advanced Analytics Module for Emotion Recognition
Provides comprehensive emotion analysis, metrics calculation, and report generation
"""

from datetime import datetime, timedelta
from collections import Counter, defaultdict
import json

# For report generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class AdvancedAnalytics:
    """Advanced analytics engine for emotion tracking"""
    
    def __init__(self):
        self.emotion_transitions = defaultdict(lambda: defaultdict(int))
        self.hourly_emotions = defaultdict(lambda: defaultdict(int))
        self.stress_indicators = []
        self.productivity_score = 0.0
        self.wellbeing_score = 0.0
        
    def calculate_wellbeing_score(self, emotion_log):
        """Calculate overall wellbeing score (0-100)"""
        if not emotion_log:
            return 50.0
        
        # Get recent emotions (last 50 entries)
        recent_emotions = emotion_log[-50:]
        
        # Emotion weights for wellbeing
        emotion_weights = {
            'happy': 100,
            'surprise': 80,
            'neutral': 60,
            'disgust': 40,
            'sad': 30,
            'fear': 20,
            'angry': 10
        }
        
        # Calculate weighted average
        total_weight = 0
        total_score = 0
        
        for entry in recent_emotions:
            emotion = entry['emotion']
            confidence = entry['confidence']
            weight = emotion_weights.get(emotion, 50)
            total_score += weight * confidence
            total_weight += confidence
        
        return (total_score / total_weight) if total_weight > 0 else 50.0
    
    def calculate_productivity_score(self, emotion_log):
        """Calculate productivity score based on emotion patterns (0-100)"""
        if not emotion_log:
            return 50.0
        
        # Get recent emotions (last 30 entries)
        recent_emotions = emotion_log[-30:]
        
        # Productivity-positive emotions
        productive_emotions = {'happy', 'neutral', 'surprise'}
        distracting_emotions = {'angry', 'sad', 'fear', 'disgust'}
        
        productive_count = sum(1 for e in recent_emotions if e['emotion'] in productive_emotions)
        distracting_count = sum(1 for e in recent_emotions if e['emotion'] in distracting_emotions)
        
        total = len(recent_emotions)
        if total == 0:
            return 50.0
        else:
            return (productive_count / total) * 100
    
    def calculate_stability_score(self, emotion_log):
        """Calculate emotional stability score (0-100)"""
        if len(emotion_log) < 10:
            return 50.0
        
        recent = emotion_log[-30:]
        emotion_changes = 0
        
        for i in range(1, len(recent)):
            if recent[i]['emotion'] != recent[i-1]['emotion']:
                emotion_changes += 1
        
        # Lower changes = higher stability
        change_rate = emotion_changes / len(recent)
        stability = 100 - (change_rate * 100)
        
        return max(0, min(100, stability))
    
    def track_emotion_transition(self, from_emotion, to_emotion):
        """Track emotion transitions"""
        if from_emotion and from_emotion != to_emotion:
            self.emotion_transitions[from_emotion][to_emotion] += 1
    
    def track_hourly_emotion(self, emotion):
        """Track emotions by hour"""
        current_hour = datetime.now().hour
        self.hourly_emotions[current_hour][emotion] += 1
    
    def add_stress_indicator(self, emotion, confidence):
        """Add stress event"""
        if emotion in ['angry', 'fear', 'sad'] and confidence > 0.7:
            self.stress_indicators.append({
                'timestamp': datetime.now().isoformat(),
                'emotion': emotion,
                'confidence': confidence
            })
    
    def get_recent_stress_count(self, hours=24):
        """Get stress indicators from last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return len([s for s in self.stress_indicators 
                   if datetime.fromisoformat(s['timestamp']) >= cutoff])
    
    def generate_insights(self, emotion_log, wellbeing_score, productivity_score):
        """Generate personalized insights"""
        insights = []
        
        if not emotion_log:
            return ["Start tracking emotions to receive personalized insights!"]
        
        # Analyze recent emotions
        recent = emotion_log[-50:] if len(emotion_log) >= 50 else emotion_log
        emotion_counts = Counter([e['emotion'] for e in recent])
        
        # Insight Most common emotion
        most_common = emotion_counts.most_common(1)[0]
        insights.append(f"Your dominant emotion recently is {most_common[0]} ({most_common[1]} occurrences).")
        
        # Insight: Wellbeing trend
        if wellbeing_score >= 75:
            insights.append("Your wellbeing score is excellent! Keep up the positive mindset.")
        elif wellbeing_score < 50:
            insights.append("Your wellbeing score suggests you could benefit from stress-relief activities.")
        
        # Insight: Productivity
        if productivity_score >= 70:
            insights.append("You're maintaining good emotional balance for productivity!")
        elif productivity_score < 40:
            insights.append("Distracting emotions detected. Consider taking short breaks to reset.")
        
        # Insight: Stress patterns
        recent_stress_count = self.get_recent_stress_count(24)
        if recent_stress_count > 10:
            insights.append("High stress levels detected. Try breathing exercises or meditation.")
        
        # Insight Emotion stability
        stability = self.calculate_stability_score(emotion_log)
        if stability < 40:
            insights.append("Your emotions have been fluctuating frequently. Consider establishing a calming routine.")
        elif stability > 70:
            insights.append("You're maintaining excellent emotional stability!")
        
        return insights
    
    def get_score_color(self, score):
        """Get color based on score value"""
        if score >= 75:
            return '#00ff88'
        elif score >= 50:
            return '#ffaa00'
        else:
            return '#ff4444'
    
    def get_wellbeing_interpretation(self, score):
        """Get wellbeing interpretation text"""
        if score >= 80:
            return "Excellent! You're in a great emotional state."
        elif score >= 60:
            return "Good! Overall positive wellbeing."
        elif score >= 40:
            return "Fair. Consider some self-care activities."
        else:
            return "Needs attention. Please take care of yourself."


class ReportGenerator:
    """Generate various report formats"""
    
    @staticmethod
    def generate_pdf_report(filename, user, emotion_log, analytics):
        """Generate comprehensive PDF report"""
        if not REPORTLAB_AVAILABLE:
            raise ImportError("reportlab is required for PDF generation. Install: pip install reportlab")
        
        doc = SimpleDocTemplate(filename, pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#6a4c93'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#8b5cf6'),
            spaceAfter=12
        )
        
        # Title
        story.append(Paragraph("🎭 Emotion Analytics Report", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # User Info
        story.append(Paragraph(f"<b>User:</b> {user}", styles['Normal']))
        story.append(Paragraph(f"<b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Paragraph(f"<b>Total Emotions Logged:</b> {len(emotion_log)}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Executive Summary
        wellbeing = analytics.calculate_wellbeing_score(emotion_log)
        productivity = analytics.calculate_productivity_score(emotion_log)
        stability = analytics.calculate_stability_score(emotion_log)
        
        story.append(Paragraph(" Executive Summary", heading_style))
        story.append(Paragraph(f"<b>Wellbeing Score:</b> {wellbeing:.1f}/100 ({analytics.get_wellbeing_interpretation(wellbeing)})", styles['Normal']))
        story.append(Paragraph(f"<b>Productivity Index:</b> {productivity:.1f}/100", styles['Normal']))
        story.append(Paragraph(f"<b>Emotional Stability:</b> {stability:.1f}/100", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Emotion Distribution
        story.append(Paragraph(" Emotion Distribution", heading_style))
        emotion_counts = Counter([e['emotion'] for e in emotion_log])
        
        table_data = [['Emotion', 'Count', 'Percentage']]
        total = len(emotion_log)
        for emotion, count in emotion_counts.most_common():
            percentage = (count / total) * 100
            table_data.append([emotion.capitalize(), str(count), f"{percentage:.1f}%"])
        
        table = Table(table_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6a4c93')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f0f0')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        story.append(Spacer(1, 0.3*inch))
        
        # Insights
        story.append(Paragraph(" Personalized Insights", heading_style))
        insights = analytics.generate_insights(emotion_log, wellbeing, productivity)
        for insight in insights:
            story.append(Paragraph(f"• {insight}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Stress Analysis
        recent_stress = analytics.get_recent_stress_count(24)
        story.append(Paragraph(" Stress Analysis (Last 24h)", heading_style))
        story.append(Paragraph(f"<b>Stress Events Detected:</b> {recent_stress}", styles['Normal']))
        
        if recent_stress > 0:
            stress_level = "Low" if recent_stress < 5 else "Moderate" if recent_stress < 15 else "High"
            story.append(Paragraph(f"<b>Stress Level:</b> {stress_level}", styles['Normal']))
        
        story.append(PageBreak())
        
        # Recent Emotion Log (last 20 entries)
        story.append(Paragraph(" Recent Emotion Log", heading_style))
        recent_log = emotion_log[-20:]
        log_data = [['Timestamp', 'Emotion', 'Confidence']]
        
        for entry in reversed(recent_log):
            timestamp = datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            emotion = entry['emotion'].capitalize()
            confidence = f"{entry['confidence']*100:.1f}%"
            log_data.append([timestamp, emotion, confidence])
        
        log_table = Table(log_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
        log_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6a4c93')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f0f0f0')),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8)
        ]))
        story.append(log_table)
        
        # Build PDF
        doc.build(story)
    
    @staticmethod
    def generate_excel_report(filename, user, emotion_log, analytics):
        """Generate Excel report with multiple sheets"""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for Excel generation. Install: pip install pandas openpyxl")
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Sheet 1: Raw Emotion Log
            df_log = pd.DataFrame(emotion_log)
            df_log['timestamp'] = pd.to_datetime(df_log['timestamp'])
            df_log.to_excel(writer, sheet_name='Emotion Log', index=False)
            
            # Sheet 2: Summary Statistics
            emotion_counts = Counter([e['emotion'] for e in emotion_log])
            df_summary = pd.DataFrame([
                {'Emotion': emotion, 'Count': count, 'Percentage': f"{(count/len(emotion_log))*100:.1f}%"}
                for emotion, count in emotion_counts.most_common()
            ])
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 3: Advanced Metrics
            wellbeing = analytics.calculate_wellbeing_score(emotion_log)
            productivity = analytics.calculate_productivity_score(emotion_log)
            stability = analytics.calculate_stability_score(emotion_log)
            stress_count = analytics.get_recent_stress_count(24)
            
            df_metrics = pd.DataFrame([
                {'Metric': 'Wellbeing Score', 'Value': f"{wellbeing:.1f}/100"},
                {'Metric': 'Productivity Index', 'Value': f"{productivity:.1f}/100"},
                {'Metric': 'Emotional Stability', 'Value': f"{stability:.1f}/100"},
                {'Metric': 'Total Emotions Logged', 'Value': len(emotion_log)},
                {'Metric': 'Stress Events (24h)', 'Value': stress_count}
            ])
            df_metrics.to_excel(writer, sheet_name='Metrics', index=False)
            
            # Sheet 4: Hourly Analysis
            if analytics.hourly_emotions:
                hourly_data = []
                for hour, emotions_dict in sorted(analytics.hourly_emotions.items()):
                    hour_12 = hour % 12 if hour % 12 != 0 else 12
                    am_pm = 'AM' if hour < 12 else 'PM'
                    time_str = f"{hour_12}:00 {am_pm}"
                    
                    for emotion, count in emotions_dict.items():
                        hourly_data.append({
                            'Hour': time_str,
                            'Emotion': emotion,
                            'Count': count
                        })
                
                if hourly_data:
                    df_hourly = pd.DataFrame(hourly_data)
                    df_hourly.to_excel(writer, sheet_name='Hourly Patterns', index=False)
            
            # Sheet 5: Insights
            insights = analytics.generate_insights(emotion_log, wellbeing, productivity)
            df_insights = pd.DataFrame([{'Insight': insight} for insight in insights])
            df_insights.to_excel(writer, sheet_name='Insights', index=False)
    
    @staticmethod
    def generate_json_report(filename, user, emotion_log, analytics):
        """Generate JSON data export"""
        wellbeing = analytics.calculate_wellbeing_score(emotion_log)
        productivity = analytics.calculate_productivity_score(emotion_log)
        stability = analytics.calculate_stability_score(emotion_log)
        stress_count = analytics.get_recent_stress_count(24)
        
        export_data = {
            'user': user,
            'report_generated': datetime.now().isoformat(),
            'metrics': {
                'wellbeing_score': round(wellbeing, 2),
                'productivity_score': round(productivity, 2),
                'stability_score': round(stability, 2),
                'total_emotions_logged': len(emotion_log),
                'stress_events_24h': stress_count
            },
            'emotion_log': emotion_log,
            'emotion_distribution': dict(Counter([e['emotion'] for e in emotion_log])),
            'stress_indicators': analytics.stress_indicators,
            'insights': analytics.generate_insights(emotion_log, wellbeing, productivity),
            'emotion_transitions': {k: dict(v) for k, v in analytics.emotion_transitions.items()},
            'hourly_patterns': {str(k): dict(v) for k, v in analytics.hourly_emotions.items()}
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
