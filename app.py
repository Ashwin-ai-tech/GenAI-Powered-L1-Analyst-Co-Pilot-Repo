# app.py - WITH REAL-TIME ANALYTICS STREAMING
from flask import Flask, request, jsonify, Response, send_file, stream_with_context, send_from_directory
from servicenow_integration import servicenow_client
import json
import os
import time
import logging
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import database

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from new_backend import (
    query_rag, 
    get_all_sessions_from_db, 
    get_conversation_history, 
    submit_feedback, 
    get_performance_stats,
    create_new_session,
    load_session,
    delete_session,
    clear_session_history,
    rename_session,
    update_theme_preference,
    get_next_word_predictions,
    analytics_manager  # Import the analytics manager
)

# --- Real-Time Analytics Features ---
analytics_executor = ThreadPoolExecutor(max_workers=8)
realtime_metrics = {}

class RealTimeAnalyticsStream:
    def __init__(self):
        self.subscribers = []
    
    def add_subscriber(self, queue):
        self.subscribers.append(queue)
    
    def remove_subscriber(self, queue):
        if queue in self.subscribers:
            self.subscribers.remove(queue)
    
    def broadcast(self, data):
        for queue in self.subscribers[:]:
            try:
                queue.put(data)
            except Exception:
                self.remove_subscriber(queue)

realtime_stream = RealTimeAnalyticsStream()

def background_metrics_updater():
    """Background thread to update real-time metrics for animations."""
    with app.app_context():  # <-- THIS IS THE FIX. It gives the thread access to the app.
        while True:
            try:
                # All the code inside this loop now runs with the correct context
                performance_data = analytics_manager.get_performance_stats()
                comprehensive_data = analytics_manager.get_comprehensive_analytics()
                
                basic_stats = comprehensive_data.get('basic_stats', {})
                system_health = comprehensive_data.get('system_health', {})
                recent_perf = performance_data.get('recent_performance', {})
                
                # Update global metrics dictionary
                realtime_metrics['total_queries'] = basic_stats.get('total_queries', 0)
                realtime_metrics['success_rate'] = basic_stats.get('success_rate', 0)
                realtime_metrics['user_satisfaction'] = basic_stats.get('satisfaction_rate', 0)
                realtime_metrics['queries_per_minute'] = recent_perf.get('queries_last_hour', 0) / 60
                realtime_metrics['avg_response_time'] = recent_perf.get('avg_response_time_ms', 0)
                realtime_metrics['system_health'] = system_health.get('health_score', 100)
                realtime_metrics['active_users'] = performance_data.get('active_sessions', 0)
                realtime_metrics['last_updated'] = datetime.now().isoformat()

                # Broadcast the new data to all connected dashboard clients
                data_to_send = {'type': 'metrics_update', 'metrics': realtime_metrics}
                realtime_stream.broadcast(json.dumps(data_to_send))
                
                time.sleep(3) # Update every 3 seconds
            except Exception as e:
                logger.error(f"Background metrics updater error: {e}")
                time.sleep(10)

# --- Flask App Setup ---
app = Flask(__name__, static_folder='static', static_url_path='')
database.init_app(app)

# Helper function to get session data
def get_session(session_id, user_id='default'):
    """Get session data - compatible with your backend"""
    try:
        return load_session(session_id, user_id)
    except Exception as e:
        logger.error(f"Error getting session {session_id}: {e}")
        return None

@app.route('/')
def serve_frontend():
    try:
        return send_from_directory('static', 'index.html')
    except Exception as e:
        return f"Error loading index.html: {str(e)}", 500

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# === REAL-TIME ANALYTICS STREAM ENDPOINT ===
@app.route('/api/analytics/stream')
def analytics_live_stream():
    def generate():
        q = Queue()
        realtime_stream.add_subscriber(q)
        try:
            # Send initial state immediately
            initial_data = {'type': 'initial_state', 'metrics': realtime_metrics}
            yield f"data: {json.dumps(initial_data)}\n\n"
            while True:
                data = q.get()
                yield f"data: {data}\n\n"
        except GeneratorExit:
            realtime_stream.remove_subscriber(q)

    return Response(generate(), mimetype='text/event-stream')

# === SERVICENOW INTEGRATION ENDPOINTS ===
@app.route('/api/servicenow/create-ticket', methods=['POST'])
def create_servicenow_ticket():
    try:
        data = request.get_json()
        
        session_id = data.get('session_id')
        short_description = data.get('short_description')
        description = data.get('description')
        urgency = data.get('urgency')
        category = data.get('category')
        use_llm_analysis = data.get('use_llm_analysis', False)
        
        # Get conversation history using the proper function
        conversation_history = []
        if session_id:
            session_history = get_conversation_history(session_id)
            if session_history:
                conversation_history = session_history
                logger.info(f"Retrieved {len(conversation_history)} messages from conversation history")
        
        # Create ticket using the form values directly
        result = servicenow_client.analyze_and_create_incident(
            conversation_history=conversation_history,
            use_llm=use_llm_analysis,
            short_description=short_description,
            description=description,
            urgency=urgency,
            category=category
        )
        
        logger.info(f"Ticket creation result: {result}")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error creating ServiceNow ticket: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Failed to create ticket: {str(e)}"
        }), 500

@app.route('/api/servicenow/get-ticket', methods=['POST'])
def get_servicenow_ticket():
    """Fetch ServiceNow ticket details by ticket number"""
    try:
        data = request.get_json()
        ticket_number = data.get('ticket_number')
        
        if not ticket_number:
            return jsonify({
                "status": "error",
                "message": "Ticket number is required"
            }), 400
        
        logger.info(f"Fetching ticket: {ticket_number}")
        
        result = servicenow_client.get_servicenow_ticket(ticket_number)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error fetching ServiceNow ticket: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Failed to fetch ticket: {str(e)}"
        }), 500

@app.route('/api/servicenow/analyze-conversation', methods=['POST'])
def analyze_conversation():
    """Preview analysis without creating ticket"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if not session_id:
            return jsonify({'error': 'Session ID is required'}), 400
        
        session_history = get_conversation_history(session_id)
        
        if not session_history:
            return jsonify({'error': 'No conversation history found'}), 400
        
        logger.info(f"Analyzing {len(session_history)} messages for ticket creation")
        
        # Try LLM analysis first
        llm_result = servicenow_client.ticket_analyzer.analyze_conversation_llm(session_history)
        
        if llm_result["status"] == "success":
            logger.info("LLM analysis successful")
            return jsonify({
                "status": "success",
                "analysis": llm_result["analysis"],
                "source": "llm"
            })
        else:
            # Fallback to rule-based
            logger.info(f"LLM failed, using rule-based: {llm_result.get('message', 'Unknown error')}")
            rule_based = servicenow_client.ticket_analyzer.analyze_conversation_rule_based(session_history)
            return jsonify({
                "status": "success", 
                "analysis": rule_based,
                "source": "rule_based"
            })
            
    except Exception as e:
        logger.error(f"Error analyzing conversation: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Analysis failed: {str(e)}'
        }), 500

@app.route('/api/servicenow/troubleshoot-ticket', methods=['POST'])
def troubleshoot_ticket():
    """Direct KB search using ServiceNow ticket details"""
    try:
        data = request.get_json()
        ticket_data = data.get('ticket_data')
        
        if not ticket_data:
            return jsonify({'error': 'Ticket data is required'}), 400
        
        # Extract relevant information from ticket
        ticket_number = ticket_data.get('number', '')
        short_description = ticket_data.get('short_description', '')
        description = ticket_data.get('description', '')
        urgency = ticket_data.get('urgency', '')
        state = ticket_data.get('state', '')
        
        # Create enhanced query for KB search
        query = f"Troubleshoot ServiceNow ticket {ticket_number}: {short_description}. {description}. Urgency: {urgency}, State: {state}"
        
        # Use your existing RAG system to search KB
        result = query_rag(query, session_id=None, user_id="default")
        
        return jsonify({
            'status': 'success',
            'ticket_number': ticket_number,
            'query': query,
            'kb_response': result
        })
        
    except Exception as e:
        logger.error(f"Error troubleshooting ticket: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Failed to troubleshoot ticket: {str(e)}'
        }), 500

@app.route('/api/servicenow/test-connection', methods=['GET'])
def test_servicenow_connection():
    """Test ServiceNow connection"""
    try:
        result = servicenow_client.test_connection()
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Connection test failed: {str(e)}'
        }), 500

# === COMPATIBLE CHAT ENDPOINT (Streaming) ===
@app.route('/api/chat', methods=['POST'])
def chat_stream():
    """Streaming chat endpoint compatible with frontend"""
    data = request.json
    query = data.get('message', '')
    session_id = data.get('session_id', 'default')
    
    if not query:
        return jsonify({'error': 'Message is required'}), 400

    def generate():
        try:
            start_time = time.time()
            
            # Broadcast query start for live feed
            realtime_stream.broadcast(json.dumps({
                'type': 'query_start', 
                'query_preview': query[:50] + '...',
                'timestamp': datetime.now().isoformat()
            }))
            
            # Get the response from your RAG system
            result = query_rag(query, session_id)
            answer = result.get("answer", "")
            
            # Record query metrics for analytics
            analytics_manager.record_query_metrics({
                'query': query,
                'confidence': result.get('confidence', 0),
                'success': result.get('used_kb', False),
                'response_time': result.get('total_time', 0),
                'kb_used': result.get('used_kb', False),
                'session_id': session_id
            })
            
            # Stream the response word by word (simulate typing)
            words = answer.split()
            for word in words:
                yield f"data: {json.dumps({'type': 'token', 'content': word + ' '})}\n\n"
                time.sleep(0.05)  # Typing speed
            
            # Signal completion with full result metadata
            yield f"data: {json.dumps({
                'type': 'complete',
                'confidence': result.get('confidence', 0),
                'used_kb': result.get('used_kb', False),
                'session_id': result.get('session_id', session_id),
                'feedback_required': result.get('feedback_required', False)
            })}\n\n"
            
            # Broadcast completion
            realtime_stream.broadcast(json.dumps({
                'type': 'query_complete', 
                'session_id': session_id,
                'response_time': time.time() - start_time
            }))
            
        except Exception as e:
            logger.error(f"Error in chat stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': 'Sorry, an error occurred.'})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/plain')

# === COMPATIBLE SESSIONS ENDPOINT ===
@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Get all sessions for sidebar - COMPATIBLE VERSION"""
    try:
        user_id = request.args.get('user_id', 'default')
        sessions = get_all_sessions_from_db(user_id)
        
        # Transform to frontend-compatible format
        compatible_sessions = []
        for session in sessions:
            compatible_sessions.append({
                'session_id': session.get('session_id'),
                'title': session.get('title', 'New Chat'),
                'created_at': session.get('created_at'),
                'last_activity': session.get('last_activity'),
                'message_count': session.get('message_count', 0),
                'theme_preference': session.get('theme_preference', 'system')
            })
        
        return jsonify(compatible_sessions)
    except Exception as e:
        logger.error(f"Error getting sessions: {e}")
        return jsonify([])

@app.route('/api/session/<session_id>', methods=['GET'])
def get_session_history(session_id):
    """Get conversation history for a specific session - COMPATIBLE VERSION"""
    try:
        user_id = request.args.get('user_id', 'default')
        
        # Try to load from backend session system first
        session_data = load_session(session_id, user_id)
        if session_data and session_data.get('conversation_history'):
            # Transform backend session format to frontend format
            history = []
            for msg in session_data['conversation_history']:
                if 'query' in msg and 'answer' in msg:
                    history.append({
                        'query': msg['query'],
                        'answer': msg['answer'],
                        'confidence': msg.get('confidence', 0),
                        'used_kb': msg.get('used_kb', False),
                        'timestamp': msg.get('timestamp', ''),
                        'clarification_asked': msg.get('clarification_asked', False)
                    })
            return jsonify({
                'session_id': session_id,
                'messages': history
            })
        
        # Fallback to history table
        history = get_conversation_history(session_id)
        return jsonify({
            'session_id': session_id,
            'messages': history
        })
            
    except Exception as e:
        logger.error(f"Error getting session history: {e}")
        return jsonify({'session_id': session_id, 'messages': []}), 500

# === COMPATIBLE FEEDBACK ENDPOINT ===
@app.route('/api/feedback', methods=['POST'])
def submit_feedback_route():
    """Submit user feedback - COMPATIBLE VERSION"""
    try:
        data = request.json
        session_id = data.get('session_id')
        query = data.get('query', '')
        answer = data.get('answer', '')
        rating = data.get('rating')
        comment = data.get('comment', '')
        
        if not session_id or rating is None:
            return jsonify({'success': False, 'error': 'Session ID and rating are required'}), 400
        
        # Convert frontend rating (0/1) to backend rating (1-5)
        backend_rating = 5 if rating == 1 else 1  # 1=thumbs down, 5=thumbs up
        
        result = submit_feedback(session_id, query, answer, backend_rating, comment)
        
        # Record feedback in analytics
        analytics_manager.record_feedback(session_id, query, answer, backend_rating, comment)
        
        return jsonify({'success': result.get('success', False)})
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        return jsonify({'success': False})

# === COMPATIBLE ANALYTICS ENDPOINT ===
@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get real analytics data from database - COMPATIBLE VERSION"""
    try:
        # Get comprehensive analytics from analytics manager
        analytics_data = analytics_manager.get_comprehensive_analytics()
        performance_stats = analytics_manager.get_performance_stats()
        
        # Transform to frontend-compatible format
        basic_stats = analytics_data.get('basic_stats', {})
        user_engagement = analytics_data.get('user_engagement', {})
        system_health = analytics_data.get('system_health', {})
        
        return jsonify({
            'total_queries': performance_stats.get('total_queries', 0),
            'success_rate': performance_stats.get('kb_usage_rate', 0),  # Using KB usage as success proxy
            'kb_usage_rate': performance_stats.get('kb_usage_rate', 0),
            'avg_confidence': performance_stats.get('avg_confidence', 0),
            'total_feedback': basic_stats.get('total_feedback', 0),
            'avg_rating': basic_stats.get('average_rating', 0),
            'total_users': performance_stats.get('active_sessions', 1),
            'messages_per_session': user_engagement.get('avg_queries_per_session', 0),
            'kb_entries': performance_stats.get('kb_entries', 0),
            'positive_rate': basic_stats.get('satisfaction_rate', 0),
            'accuracy_rate': performance_stats.get('kb_usage_rate', 0),  # Using KB usage as accuracy proxy
            'system_health': system_health.get('health_score', 0)
        })
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        return jsonify({
            'total_queries': 0,
            'success_rate': 0,
            'kb_usage_rate': 0,
            'avg_confidence': 0,
            'total_feedback': 0,
            'avg_rating': 0,
            'total_users': 1,
            'messages_per_session': 0,
            'kb_entries': 0,
            'positive_rate': 0,
            'accuracy_rate': 0,
            'system_health': 0
        })

# === NEW SESSION MANAGEMENT ENDPOINTS ===
@app.route('/api/sessions/new', methods=['POST'])
def create_new_session_route():
    """Create a new session"""
    try:
        data = request.json or {}
        user_id = data.get('user_id', 'default')
        
        session_info = create_new_session(user_id)
        return jsonify(session_info)
        
    except Exception as e:
        logger.error(f"Error creating new session: {e}")
        return jsonify({'error': 'Failed to create session'}), 500

@app.route('/api/sessions/<session_id>/delete', methods=['POST'])
def delete_session_route(session_id):
    """Delete a session"""
    try:
        user_id = request.args.get('user_id', 'default')
        success = delete_session(session_id, user_id)
        
        return jsonify({'success': success})
        
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        return jsonify({'success': False}), 500

@app.route('/api/sessions/<session_id>/clear', methods=['POST'])
def clear_session_route(session_id):
    """Clear session history"""
    try:
        user_id = request.args.get('user_id', 'default')
        success = clear_session_history(session_id, user_id)
        
        return jsonify({'success': success})
        
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        return jsonify({'success': False}), 500

@app.route('/api/sessions/<session_id>/rename', methods=['POST'])
def rename_session_route(session_id):
    """Rename a session"""
    try:
        data = request.json
        new_title = data.get('new_title')
        user_id = data.get('user_id', 'default')
        
        if not new_title:
            return jsonify({'error': 'New title is required'}), 400
            
        success = rename_session(session_id, new_title, user_id)
        return jsonify({'success': success})
        
    except Exception as e:
        logger.error(f"Error renaming session: {e}")
        return jsonify({'success': False}), 500

@app.route('/api/sessions/<session_id>/theme', methods=['POST'])
def update_theme_route(session_id):
    """Update theme preference"""
    try:
        data = request.json
        theme = data.get('theme')
        user_id = data.get('user_id', 'default')
        
        if not theme:
            return jsonify({'error': 'Theme is required'}), 400
            
        success = update_theme_preference(session_id, theme, user_id)
        return jsonify({'success': success})
        
    except Exception as e:
        logger.error(f"Error updating theme: {e}")
        return jsonify({'success': False}), 500

# === HEALTH CHECK ===
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'version': '1.0'
    })

# === QUERY ENDPOINT (Backward Compatibility) ===
@app.route('/api/query', methods=['POST'])
def api_query():
    """Original query endpoint for backward compatibility"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        session_id = data.get('session_id')
        user_id = data.get('user_id', 'default')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        result = query_rag(query, session_id, user_id)
        
        # Record query metrics for analytics
        analytics_manager.record_query_metrics({
            'query': query,
            'confidence': result.get('confidence', 0),
            'success': result.get('used_kb', False),
            'response_time': result.get('total_time', 0),
            'kb_used': result.get('used_kb', False),
            'session_id': session_id or result.get('session_id')
        })
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API query error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# === PREDICTIONS ENDPOINT ===
@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    """Get next word predictions"""
    try:
        partial_text = request.args.get('partial_text', '')
        predictions = get_next_word_predictions(partial_text)
        
        return jsonify({'predictions': predictions})
        
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        return jsonify({'predictions': []})

# === ENHANCED ANALYTICS DETAILS ===
@app.route('/api/analytics/details', methods=['GET'])
def get_detailed_analytics():
    """Get detailed analytics for admin panel"""
    try:
        # Get comprehensive analytics from analytics manager
        analytics_data = analytics_manager.get_comprehensive_analytics()
        performance_stats = analytics_manager.get_performance_stats()
        trend_analysis = analytics_manager.get_trend_analysis(days=30)
        
        basic_stats = analytics_data.get('basic_stats', {})
        query_trends = analytics_data.get('query_trends', {})
        daily_metrics = analytics_data.get('daily_metrics', [])
        user_engagement = analytics_data.get('user_engagement', {})
        system_health = analytics_data.get('system_health', {})
        
        return jsonify({
            'basic_stats': {
                'total_queries': performance_stats.get('total_queries', 0),
                'success_rate': performance_stats.get('kb_usage_rate', 0),
                'kb_usage_rate': performance_stats.get('kb_usage_rate', 0),
                'avg_confidence': performance_stats.get('avg_confidence', 0),
                'total_feedback': basic_stats.get('total_feedback', 0),
                'avg_rating': basic_stats.get('average_rating', 0),
                'satisfaction_rate': basic_stats.get('satisfaction_rate', 0)
            },
            'query_analytics': {
                'top_queries': query_trends.get('top_queries', []),
                'categories': query_trends.get('categories', {}),
                'daily_trends': daily_metrics
            },
            'user_analytics': {
                'total_users': performance_stats.get('active_sessions', 1),
                'avg_queries_per_session': user_engagement.get('avg_queries_per_session', 0),
                'avg_session_duration': user_engagement.get('avg_session_duration_minutes', 0),
                'feedback_rate': user_engagement.get('feedback_rate', 0)
            },
            'performance_analytics': {
                'avg_response_time': system_health.get('avg_total_time_ms', 0),
                'avg_retrieval_time': system_health.get('avg_retrieval_time_ms', 0),
                'error_rate': system_health.get('error_rate', 0),
                'health_score': system_health.get('health_score', 0),
                'active_sessions': system_health.get('avg_active_sessions', 0)
            },
            'trend_analysis': trend_analysis,
            'rating_distribution': analytics_data.get('rating_distribution', {}),
            'recent_feedback': analytics_data.get('recent_feedback', [])
        })
    except Exception as e:
        logger.error(f"Error getting detailed analytics: {e}")
        return jsonify({}), 500

# === TREND ANALYSIS ENDPOINT ===
@app.route('/api/analytics/trends', methods=['GET'])
def get_trend_analysis():
    """Get trend analysis for specified period"""
    try:
        days = request.args.get('days', 30, type=int)
        trend_data = analytics_manager.get_trend_analysis(days=days)
        return jsonify(trend_data)
    except Exception as e:
        logger.error(f"Error getting trend analysis: {e}")
        return jsonify({'error': 'Failed to get trend analysis'}), 500

# === SYSTEM PERFORMANCE ENDPOINT ===
@app.route('/api/analytics/performance', methods=['GET'])
def get_system_performance():
    """Get detailed system performance metrics"""
    try:
        performance_stats = analytics_manager.get_performance_stats()
        analytics_data = analytics_manager.get_comprehensive_analytics()
        system_health = analytics_data.get('system_health', {})
        
        return jsonify({
            'performance_stats': performance_stats,
            'system_health': system_health,
            'recent_performance': performance_stats.get('recent_performance', {})
        })
    except Exception as e:
        logger.error(f"Error getting system performance: {e}")
        return jsonify({'error': 'Failed to get performance data'}), 500

# === QUERY PATTERNS ENDPOINT ===
@app.route('/api/analytics/query-patterns', methods=['GET'])
def get_query_patterns():
    """Get query pattern analysis"""
    try:
        analytics_data = analytics_manager.get_comprehensive_analytics()
        query_trends = analytics_data.get('query_trends', {})
        
        return jsonify({
            'top_queries': query_trends.get('top_queries', []),
            'categories': query_trends.get('categories', {}),
            'total_patterns': len(query_trends.get('top_queries', []))
        })
    except Exception as e:
        logger.error(f"Error getting query patterns: {e}")
        return jsonify({'error': 'Failed to get query patterns'}), 500

# === ANALYTICS DASHBOARD ENDPOINTS ===
@app.route('/analytics')
def serve_analytics():
    """Serve the analytics dashboard"""
    try:
        return send_from_directory('analytics', 'an_index.html')
    except Exception as e:
        return f"Error loading analytics dashboard: {str(e)}", 500

@app.route('/analytics/<path:path>')
def serve_analytics_static(path):
    """Serve static files for analytics"""
    return send_from_directory('analytics', path)

@app.route('/api/analytics/user', methods=['GET'])
def get_user_analytics():
    """Get user analytics data"""
    try:
        user_id = request.args.get('user_id', 'default')
        user_dashboard = analytics_manager.get_user_dashboard(user_id)
        return jsonify(user_dashboard)
    except Exception as e:
        logger.error(f"Error getting user analytics: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/analytics/health', methods=['GET'])
def get_system_health():
    """Get system health data"""
    try:
        admin_dashboard = analytics_manager.get_admin_dashboard()
        return jsonify(admin_dashboard.get('system_health', {}))
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/analytics/export', methods=['POST'])
def export_analytics():
    """Export analytics report"""
    try:
        report_type = request.json.get('type', 'comprehensive')
        filename = analytics_manager.export_analytics_report(report_type)
        
        if filename:
            return send_file(filename, as_attachment=True)
        else:
            return jsonify({'error': 'Export failed'}), 500
            
    except Exception as e:
        logger.error(f"Error exporting analytics: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# === CLEANUP ENDPOINT (Admin only) ===
@app.route('/api/admin/cleanup-analytics', methods=['POST'])
def cleanup_analytics():
    """Clean up old analytics data (admin only)"""
    try:
        analytics_manager.cleanup_old_data()
        return jsonify({'success': True, 'message': 'Analytics cleanup completed'})
    except Exception as e:
        logger.error(f"Error during analytics cleanup: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# === RECORD PERFORMANCE METRICS ENDPOINT ===
@app.route('/api/analytics/record-metrics', methods=['POST'])
def record_performance_metrics():
    """Record system performance metrics"""
    try:
        data = request.json
        metrics = {
            'retrieval_time': data.get('retrieval_time', 0),
            'generation_time': data.get('generation_time', 0),
            'total_time': data.get('total_time', 0),
            'active_sessions': data.get('active_sessions', 0)
        }
        
        analytics_manager.record_performance_metrics(metrics)
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error recording performance metrics: {e}")
        return jsonify({'success': False}), 500

# ===============================================
# Initialize Database within Application Context
# ===============================================
with app.app_context():
    # Initialize tables from new_backend.py
    from new_backend import init_db
    init_db()
    
    # Initialize tables from analytics_manager.py
    analytics_manager.init_tables()

# Start the background thread for real-time metric updates
metrics_thread = threading.Thread(target=background_metrics_updater, daemon=True)
metrics_thread.start()

if __name__ == '__main__':
    # Check if static files exist
    static_files = ['index.html', 'style.css', 'script.js']
    for file in static_files:
        file_path = os.path.join('static', file)
        if os.path.exists(file_path):
            logger.info(f"✅ Found: {file_path}")
        else:
            logger.warning(f"❌ Missing: {file_path}")
    
    logger.info("🚀 Starting Astra IT Support Co-Pilot with Real-Time Analytics...")
    logger.info("📡 API Server running on http://localhost:5000")
    logger.info("🔗 Frontend available at http://localhost:5000")
    logger.info("📊 Enhanced Analytics System: ACTIVE")
    logger.info("🔄 Real-Time Analytics Stream: ACTIVE")
    
    app.run(debug=True, port=5000, host='0.0.0.0')
