-- SQL Analytics Queries for Smart Traffic Management System
-- Comprehensive queries for performance analysis and reporting

-- ===================================================================
-- BASIC PERFORMANCE METRICS
-- ===================================================================

-- 1. Average travel time by vehicle type
SELECT 
    vehicle_type,
    COUNT(*) as trip_count,
    AVG(duration) as avg_travel_time_sec,
    AVG(waiting_time) as avg_waiting_time_sec,
    AVG(time_loss) as avg_time_loss_sec,
    AVG(duration - time_loss) as avg_productive_time_sec
FROM trip_info
WHERE duration > 0
GROUP BY vehicle_type
ORDER BY avg_travel_time_sec DESC;

-- 2. Traffic flow by time of day (hourly breakdown)
SELECT 
    CAST(departure_time / 3600 AS INTEGER) as hour,
    COUNT(*) as vehicles_departed,
    AVG(duration) as avg_travel_time,
    AVG(route_length) as avg_distance
FROM trip_info
GROUP BY hour
ORDER BY hour;

-- 3. Lane occupancy and congestion analysis
SELECT 
    lane_id,
    AVG(occupancy) as avg_occupancy,
    MAX(occupancy) as max_occupancy,
    AVG(vehicle_count) as avg_vehicles,
    AVG(average_speed) as avg_speed_mps,
    COUNT(CASE WHEN congestion_level = 'HIGH' THEN 1 END) * 100.0 / COUNT(*) as high_congestion_pct
FROM lane_metrics
GROUP BY lane_id
HAVING avg_occupancy > 0.3
ORDER BY avg_occupancy DESC;

-- ===================================================================
-- SIGNAL PERFORMANCE ANALYSIS
-- ===================================================================

-- 4. Signal adaptation effectiveness
SELECT 
    junction_id,
    COUNT(*) as adaptation_count,
    AVG(new_duration) as avg_new_duration,
    AVG(occupancy) as avg_occupancy_at_adaptation,
    COUNT(CASE WHEN action = 'EXTEND_GREEN' THEN 1 END) as green_extensions,
    COUNT(CASE WHEN action = 'EMERGENCY_PRIORITY' THEN 1 END) as emergency_overrides
FROM signal_adaptations
GROUP BY junction_id
ORDER BY adaptation_count DESC;

-- 5. Junction delay analysis
SELECT 
    im.junction_id,
    AVG(im.average_delay) as avg_delay_sec,
    AVG(im.queue_length) as avg_queue_length,
    AVG(im.throughput) as avg_throughput_vph,
    im.level_of_service,
    COUNT(*) as measurement_count
FROM intersection_metrics im
GROUP BY im.junction_id, im.level_of_service
ORDER BY avg_delay_sec DESC;

-- ===================================================================
-- EMERGENCY VEHICLE ANALYSIS
-- ===================================================================

-- 6. Emergency vehicle response times
SELECT 
    ee.vehicle_id,
    MIN(CASE WHEN ee.event_type = 'DETECTED' THEN ee.timestamp END) as detection_time,
    MAX(CASE WHEN ee.event_type = 'COMPLETED' THEN ee.timestamp END) as completion_time,
    MAX(CASE WHEN ee.event_type = 'COMPLETED' THEN ee.response_time END) as total_response_time_sec,
    SUM(ee.halted_vehicles) as total_vehicles_halted,
    AVG(ee.speed) as avg_speed_mps
FROM emergency_events ee
GROUP BY ee.vehicle_id
ORDER BY total_response_time_sec;

-- 7. Emergency priority effectiveness
SELECT 
    event_type,
    COUNT(*) as event_count,
    AVG(halted_vehicles) as avg_halted_vehicles,
    AVG(response_time) as avg_response_time
FROM emergency_events
WHERE event_type IN ('GREEN_WAVE', 'STATUS_UPDATE', 'COMPLETED')
GROUP BY event_type;

-- ===================================================================
-- V2X COMMUNICATION ANALYSIS
-- ===================================================================

-- 8. Communication performance by message type
SELECT 
    message_type,
    COUNT(*) as message_count,
    AVG(latency) * 1000 as avg_latency_ms,
    MIN(latency) * 1000 as min_latency_ms,
    MAX(latency) * 1000 as max_latency_ms,
    AVG(distance) as avg_distance_m,
    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate_pct
FROM messages
GROUP BY message_type
ORDER BY message_count DESC;

-- 9. V2V vs V2I message distribution
SELECT 
    CASE 
        WHEN message_type LIKE 'V2V%' THEN 'V2V'
        WHEN message_type LIKE 'V2I%' THEN 'V2I'
        WHEN message_type LIKE 'I2V%' THEN 'I2V'
        ELSE 'OTHER'
    END as communication_type,
    COUNT(*) as message_count,
    AVG(latency) * 1000 as avg_latency_ms,
    AVG(distance) as avg_distance_m
FROM messages
GROUP BY communication_type;

-- 10. Communication density over time
SELECT 
    CAST(timestamp / 300 AS INTEGER) * 300 as time_bucket_sec,
    COUNT(*) as messages_in_5min,
    COUNT(DISTINCT sender_id) as active_senders,
    COUNT(DISTINCT receiver_id) as active_receivers,
    AVG(latency) * 1000 as avg_latency_ms
FROM messages
GROUP BY time_bucket_sec
ORDER BY time_bucket_sec;

-- ===================================================================
-- TRUST AND SOCIAL NETWORK ANALYSIS
-- ===================================================================

-- 11. Trust score distribution
SELECT 
    relationship_type,
    COUNT(*) as relationship_count,
    AVG(trust_score) as avg_trust,
    MIN(trust_score) as min_trust,
    MAX(trust_score) as max_trust,
    AVG(interaction_count) as avg_interactions,
    COUNT(CASE WHEN trust_score > 0.7 THEN 1 END) * 100.0 / COUNT(*) as high_trust_pct
FROM trust_scores
GROUP BY relationship_type;

-- 12. Trust evolution over time
SELECT 
    CAST(timestamp / 600 AS INTEGER) * 600 as time_bucket_sec,
    AVG(trust_score) as avg_trust_score,
    COUNT(*) as trust_updates,
    COUNT(DISTINCT entity1) as unique_entities
FROM trust_scores
GROUP BY time_bucket_sec
ORDER BY time_bucket_sec;

-- 13. Most trusted entities
SELECT 
    entity1,
    COUNT(DISTINCT entity2) as connection_count,
    AVG(trust_score) as avg_trust_received,
    SUM(interaction_count) as total_interactions
FROM trust_scores
GROUP BY entity1
HAVING avg_trust_received > 0.6
ORDER BY avg_trust_received DESC, connection_count DESC
LIMIT 20;

-- ===================================================================
-- EMISSIONS AND ENVIRONMENTAL IMPACT
-- ===================================================================

-- 14. Emissions by vehicle type
SELECT 
    vehicle_type,
    COUNT(DISTINCT vehicle_id) as vehicle_count,
    AVG(co2_emission) as avg_co2_per_timestep,
    SUM(co2_emission) as total_co2,
    AVG(fuel_consumption) as avg_fuel_per_timestep,
    SUM(fuel_consumption) as total_fuel
FROM vehicle_logs
WHERE co2_emission IS NOT NULL
GROUP BY vehicle_type
ORDER BY total_co2 DESC;

-- 15. Emissions correlation with speed and waiting time
SELECT 
    CASE 
        WHEN speed < 5 THEN '0-5 m/s'
        WHEN speed < 10 THEN '5-10 m/s'
        WHEN speed < 15 THEN '10-15 m/s'
        WHEN speed < 20 THEN '15-20 m/s'
        ELSE '20+ m/s'
    END as speed_range,
    COUNT(*) as sample_count,
    AVG(co2_emission) as avg_co2,
    AVG(fuel_consumption) as avg_fuel,
    AVG(waiting_time) as avg_waiting_time
FROM vehicle_logs
WHERE co2_emission IS NOT NULL
GROUP BY speed_range
ORDER BY avg_co2 DESC;

-- ===================================================================
-- COMPARATIVE SCENARIO ANALYSIS
-- ===================================================================

-- 16. Scenario performance comparison
SELECT 
    scenario_name,
    metric_name,
    metric_value,
    unit
FROM scenario_results
WHERE metric_name IN ('avg_travel_time', 'avg_waiting_time', 'throughput', 'avg_co2_emission')
ORDER BY scenario_name, metric_name;

-- 17. Improvement calculation (requires baseline)
SELECT 
    sr1.metric_name,
    sr1.metric_value as baseline_value,
    sr2.metric_value as smart_value,
    ((sr1.metric_value - sr2.metric_value) / sr1.metric_value * 100) as improvement_pct,
    sr1.unit
FROM scenario_results sr1
JOIN scenario_results sr2 ON sr1.metric_name = sr2.metric_name
WHERE sr1.scenario_name = 'Baseline' 
  AND sr2.scenario_name = 'Smart System'
  AND sr1.metric_value > 0;

-- ===================================================================
-- ADVANCED ANALYTICS
-- ===================================================================

-- 18. Peak hour identification
SELECT 
    CAST(timestamp / 3600 AS INTEGER) as hour,
    AVG(occupancy) as avg_occupancy,
    AVG(vehicle_count) as avg_vehicles,
    AVG(average_speed) as avg_speed,
    COUNT(CASE WHEN congestion_level = 'HIGH' THEN 1 END) as high_congestion_count
FROM lane_metrics
GROUP BY hour
HAVING avg_occupancy > 0.5
ORDER BY avg_occupancy DESC;

-- 19. Network efficiency score
SELECT 
    AVG(speed) / MAX(speed) * 100 as speed_efficiency_pct,
    AVG(CASE WHEN waiting_time = 0 THEN 1 ELSE 0 END) * 100 as zero_wait_pct,
    COUNT(DISTINCT vehicle_id) as vehicles_observed,
    AVG(duration) as avg_trip_duration
FROM trip_info
JOIN vehicle_logs ON trip_info.vehicle_id = vehicle_logs.vehicle_id;

-- 20. Bottleneck detection
SELECT 
    lane_id,
    AVG(occupancy) as avg_occupancy,
    AVG(vehicle_count) as avg_vehicles,
    MIN(average_speed) as min_speed_observed,
    SUM(total_waiting_time) as cumulative_waiting_time,
    COUNT(*) as measurement_count
FROM lane_metrics
WHERE occupancy > 0.7
GROUP BY lane_id
ORDER BY cumulative_waiting_time DESC
LIMIT 10;
