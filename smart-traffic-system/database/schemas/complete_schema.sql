-- SQLite Database Schema for Smart Traffic Management System
-- Complete schema for all logging and analytics tables

-- ===================================================================
-- TRAFFIC EVENTS AND METRICS
-- ===================================================================

-- Vehicle movement and state logs
CREATE TABLE IF NOT EXISTS vehicle_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    vehicle_id TEXT NOT NULL,
    vehicle_type TEXT,
    position_x REAL,
    position_y REAL,
    speed REAL,
    lane_id TEXT,
    edge_id TEXT,
    waiting_time REAL,
    co2_emission REAL,
    fuel_consumption REAL,
    INDEX idx_vehicle_time (vehicle_id, timestamp)
);

-- Lane occupancy and congestion metrics
CREATE TABLE IF NOT EXISTS lane_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    lane_id TEXT NOT NULL,
    occupancy REAL,
    vehicle_count INTEGER,
    average_speed REAL,
    total_waiting_time REAL,
    congestion_level TEXT,
    INDEX idx_lane_time (lane_id, timestamp)
);

-- Traffic signal state changes
CREATE TABLE IF NOT EXISTS signal_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    junction_id TEXT NOT NULL,
    phase INTEGER,
    state TEXT,
    duration REAL,
    vehicles_waiting INTEGER,
    reason TEXT,
    INDEX idx_junction_time (junction_id, timestamp)
);

-- ===================================================================
-- ADAPTIVE SIGNAL CONTROL
-- ===================================================================

-- Signal adaptation events (from adaptive_signals.py)
CREATE TABLE IF NOT EXISTS signal_adaptations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    junction_id TEXT NOT NULL,
    lane_id TEXT,
    occupancy REAL,
    current_phase INTEGER,
    action TEXT,
    new_duration INTEGER,
    reason TEXT,
    INDEX idx_adaptation_time (timestamp)
);

-- ===================================================================
-- EMERGENCY VEHICLE MANAGEMENT
-- ===================================================================

-- Emergency vehicle events and priority actions
CREATE TABLE IF NOT EXISTS emergency_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    vehicle_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    junction_id TEXT,
    lane_id TEXT,
    speed REAL,
    halted_vehicles INTEGER,
    response_time REAL,
    notes TEXT,
    INDEX idx_emergency_time (vehicle_id, timestamp)
);

-- ===================================================================
-- V2X COMMUNICATION
-- ===================================================================

-- V2V and V2I message logs
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    sender_id TEXT NOT NULL,
    receiver_id TEXT,
    message_type TEXT NOT NULL,
    content TEXT,
    latency REAL,
    distance REAL,
    success INTEGER,
    INDEX idx_message_time (timestamp),
    INDEX idx_message_type (message_type)
);

-- Communication statistics snapshots
CREATE TABLE IF NOT EXISTS communication_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    stat_type TEXT NOT NULL,
    value REAL,
    INDEX idx_stat_time (timestamp)
);

-- ===================================================================
-- SIOT TRUST MANAGEMENT
-- ===================================================================

-- Trust scores between entities
CREATE TABLE IF NOT EXISTS trust_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    entity1 TEXT NOT NULL,
    entity2 TEXT NOT NULL,
    trust_score REAL NOT NULL,
    relationship_type TEXT,
    interaction_count INTEGER,
    INDEX idx_trust_entities (entity1, entity2),
    INDEX idx_trust_time (timestamp)
);

-- Trust-related events and interactions
CREATE TABLE IF NOT EXISTS trust_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    entity1 TEXT NOT NULL,
    entity2 TEXT NOT NULL,
    event_type TEXT NOT NULL,
    trust_change REAL,
    reason TEXT,
    INDEX idx_trust_event_time (timestamp)
);

-- ===================================================================
-- TRIP AND PERFORMANCE METRICS
-- ===================================================================

-- Completed trip information
CREATE TABLE IF NOT EXISTS trip_info (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id TEXT PRIMARY KEY,
    departure_time REAL,
    arrival_time REAL,
    duration REAL,
    route_length REAL,
    waiting_time REAL,
    time_loss REAL,
    depart_delay REAL,
    vehicle_type TEXT,
    INDEX idx_trip_time (departure_time)
);

-- Intersection performance metrics
CREATE TABLE IF NOT EXISTS intersection_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    junction_id TEXT NOT NULL,
    throughput INTEGER,
    average_delay REAL,
    queue_length INTEGER,
    level_of_service TEXT,
    INDEX idx_intersection_time (junction_id, timestamp)
);

-- ===================================================================
-- SCENARIO COMPARISON
-- ===================================================================

-- Scenario comparison results
CREATE TABLE IF NOT EXISTS scenario_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario_name TEXT NOT NULL,
    run_timestamp REAL NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL,
    unit TEXT,
    notes TEXT
);

-- ===================================================================
-- VIEWS FOR ANALYTICS
-- ===================================================================

-- Average vehicle metrics by type
CREATE VIEW IF NOT EXISTS avg_metrics_by_type AS
SELECT 
    vehicle_type,
    COUNT(DISTINCT vehicle_id) as vehicle_count,
    AVG(speed) as avg_speed,
    AVG(waiting_time) as avg_waiting_time,
    AVG(co2_emission) as avg_co2,
    AVG(fuel_consumption) as avg_fuel
FROM vehicle_logs
GROUP BY vehicle_type;

-- Congestion hotspots
CREATE VIEW IF NOT EXISTS congestion_hotspots AS
SELECT 
    lane_id,
    AVG(occupancy) as avg_occupancy,
    AVG(vehicle_count) as avg_vehicles,
    COUNT(*) as measurement_count,
    SUM(CASE WHEN congestion_level = 'HIGH' THEN 1 ELSE 0 END) as high_congestion_count
FROM lane_metrics
GROUP BY lane_id
HAVING avg_occupancy > 0.6
ORDER BY avg_occupancy DESC;

-- Emergency response performance
CREATE VIEW IF NOT EXISTS emergency_performance AS
SELECT 
    vehicle_id,
    MIN(CASE WHEN event_type = 'DETECTED' THEN timestamp END) as start_time,
    MAX(CASE WHEN event_type = 'COMPLETED' THEN timestamp END) as end_time,
    MAX(CASE WHEN event_type = 'COMPLETED' THEN response_time END) as total_response_time,
    SUM(halted_vehicles) as total_halted_vehicles,
    COUNT(*) as event_count
FROM emergency_events
GROUP BY vehicle_id;

-- Communication efficiency
CREATE VIEW IF NOT EXISTS communication_efficiency AS
SELECT 
    message_type,
    COUNT(*) as message_count,
    AVG(latency) * 1000 as avg_latency_ms,
    AVG(distance) as avg_distance,
    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
FROM messages
GROUP BY message_type;

-- Trust network statistics
CREATE VIEW IF NOT EXISTS trust_network_stats AS
SELECT 
    relationship_type,
    COUNT(*) as relationship_count,
    AVG(trust_score) as avg_trust_score,
    MIN(trust_score) as min_trust_score,
    MAX(trust_score) as max_trust_score,
    AVG(interaction_count) as avg_interactions
FROM trust_scores
GROUP BY relationship_type;

-- ===================================================================
-- INDEXES FOR PERFORMANCE
-- ===================================================================

-- Additional indexes for common queries
CREATE INDEX IF NOT EXISTS idx_vehicle_type ON vehicle_logs(vehicle_type);
CREATE INDEX IF NOT EXISTS idx_lane_congestion ON lane_metrics(congestion_level);
CREATE INDEX IF NOT EXISTS idx_emergency_type ON emergency_events(event_type);
CREATE INDEX IF NOT EXISTS idx_message_sender ON messages(sender_id);
CREATE INDEX IF NOT EXISTS idx_trust_type ON trust_scores(relationship_type);
