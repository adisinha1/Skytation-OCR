import React, { useState, useCallback, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Alert,
  RefreshControl,
  Image,
} from 'react-native';
import { useFocusEffect, useRouter } from 'expo-router';

const BACKEND_URL = 'http://10.0.0.66:8000';

interface Event {
  id: number;
  plate_text: string;
  state?: string;
  confidence: number;
  timestamp: string;
  location: string;
  result: string;
  notes?: string;
  source?: string;
  time_limit_minutes?: number;
  lot_name?: string;
  has_image?: boolean;
  image_data?: string;
}

interface TimedStay {
  id: number;
  plate_text: string;
  first_seen: string;
  last_seen: string;
  time_limit_minutes: number;
  lot_name?: string;
}

interface Violation {
  id: number;
  plate_text: string;
  timestamp: string;
  location: string;
  reason: string;
  event_id?: number;
}

export default function EnforcementScreen() {
  const router = useRouter();
  const [events, setEvents] = useState<Event[]>([]);
  const [timedStays, setTimedStays] = useState<TimedStay[]>([]);
  const [violations, setViolations] = useState<Violation[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [showAllEvents, setShowAllEvents] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [eventImages, setEventImages] = useState<Record<number, string>>({});

  const loadData = async () => {
    try {
      setLoadError(null);
      
      const [eventsRes, timedRes, violationsRes] = await Promise.all([
        fetch(`${BACKEND_URL}/api/events`),
        fetch(`${BACKEND_URL}/api/timed_stays`),
        fetch(`${BACKEND_URL}/api/violations`),
      ]);

      if (eventsRes.ok) {
        const eventsData = await eventsRes.json();
        setEvents(eventsData);
        
        // Load images for ALL events that have them
        const imagesWithData = eventsData.filter((e: Event) => e.has_image);
        
        // Load images in parallel
        const imagePromises = imagesWithData.map(async (event: Event) => {
          try {
            const response = await fetch(`${BACKEND_URL}/api/events/${event.id}`);
            if (response.ok) {
              const data = await response.json();
              if (data.image_data) {
                return { id: event.id, data: data.image_data };
              }
            }
          } catch (error) {
            console.error(`Error loading image for event ${event.id}:`, error);
          }
          return null;
        });
        
        const imageResults = await Promise.all(imagePromises);
        const newImages: Record<number, string> = {};
        imageResults.forEach(result => {
          if (result) {
            newImages[result.id] = result.data;
          }
        });
        setEventImages(newImages);
      } else {
        setLoadError(`Failed to load events: ${eventsRes.status}`);
      }
      
      if (timedRes.ok) {
        setTimedStays(await timedRes.json());
      }
      
      if (violationsRes.ok) {
        const violationsData = await violationsRes.json();
        // Filter out low_confidence violations - those are not real violations
        const realViolations = violationsData.filter(
          (v: Violation) => v.reason !== 'low_confidence'
        );
        setViolations(realViolations);
      }
    } catch (error) {
      setLoadError(`Connection error: ${String(error)}`);
    }
  };

  const expireOverstays = async () => {
    try {
      await fetch(`${BACKEND_URL}/api/timed/expire`, {
        method: 'POST',
      });
    } catch (error) {
      // Silent fail - this runs in background
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await expireOverstays();
    await loadData();
    setRefreshing(false);
  };

  useFocusEffect(
    useCallback(() => {
      expireOverstays().then(() => loadData());
    }, [])
  );

  // Auto-expire check every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      expireOverstays().then(() => loadData());
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  // Update display every second for countdown
  useEffect(() => {
    const interval = setInterval(() => {
      setTimedStays(stays => [...stays]);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  // Utility functions
  const parseTimestamp = (timestamp: string): Date => {
    if (!timestamp) return new Date();
    let ts = timestamp;
    const hasTimezone = ts.endsWith('Z') || ts.includes('+') || (ts.includes('-') && ts.lastIndexOf('-') > 10);
    if (!hasTimezone && ts.includes('T')) {
      ts = ts + 'Z';
    }
    return new Date(ts);
  };

  const formatTimestamp = (timestamp: string) => {
    try {
      const date = parseTimestamp(timestamp);
      return date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        hour12: true,
      });
    } catch {
      return timestamp;
    }
  };

  const calculateTimeRemaining = (firstSeen: string, limitMinutes: number = 120) => {
    const start = parseTimestamp(firstSeen);
    const now = new Date();
    const elapsedMinutes = (now.getTime() - start.getTime()) / 1000 / 60;
    return limitMinutes - elapsedMinutes;
  };

  const formatTimeRemaining = (minutes: number) => {
    const absMinutes = Math.abs(minutes);
    if (absMinutes < 1) {
      const seconds = Math.floor(absMinutes * 60);
      return minutes < 0 ? `+${seconds}s over` : `${seconds}s`;
    }
    if (absMinutes < 60) {
      return minutes < 0 ? `+${Math.floor(absMinutes)}m over` : `${Math.floor(absMinutes)}m`;
    }
    const hours = Math.floor(absMinutes / 60);
    const mins = Math.floor(absMinutes % 60);
    const timeStr = `${hours}h ${mins}m`;
    return minutes < 0 ? `+${timeStr} over` : timeStr;
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.85) return '#4CAF50';
    if (confidence > 0.6) return '#FFA500';
    return '#F44336';
  };

  const handleEventClick = (event: Event) => {
    router.push({
      pathname: '/edit-event',
      params: { eventId: event.id.toString() }
    });
  };

  const deleteViolation = async (violationId: number) => {
    Alert.alert(
      'Delete Violation',
      'Are you sure you want to remove this violation?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              const response = await fetch(`${BACKEND_URL}/api/violations/${violationId}`, {
                method: 'DELETE',
              });
              if (response.ok) {
                await loadData();
              } else {
                Alert.alert('Error', 'Failed to delete violation');
              }
            } catch (error) {
              Alert.alert('Error', 'Failed to delete violation');
            }
          },
        },
      ]
    );
  };

  const clearTimedStay = async (stayId: number) => {
    Alert.alert(
      'Clear Parking',
      'Mark this vehicle as departed?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear',
          onPress: async () => {
            try {
              const response = await fetch(`${BACKEND_URL}/api/timed_stays/${stayId}`, {
                method: 'DELETE',
              });
              if (response.ok) {
                await loadData();
              }
            } catch (error) {
              Alert.alert('Error', 'Failed to clear parking');
            }
          },
        },
      ]
    );
  };

  const displayedEvents = showAllEvents ? events : events.slice(0, 5);

  return (
    <ScrollView
      style={styles.container}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
    >
      <View style={styles.header}>
        <Text style={styles.title}>Parking Enforcement</Text>
        <Text style={styles.subtitle}>Monitor violations and active parking</Text>
      </View>

      {loadError && (
        <View style={styles.errorCard}>
          <Text style={styles.errorText}>{loadError}</Text>
          <TouchableOpacity style={styles.retryButton} onPress={loadData}>
            <Text style={styles.retryButtonText}>Retry</Text>
          </TouchableOpacity>
        </View>
      )}

      {/* Section 1: Active Parking */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Active Parking ({timedStays.length})</Text>
        
        {timedStays.length === 0 ? (
          <Text style={styles.emptyText}>No active parked vehicles</Text>
        ) : (
          <View>
            {timedStays.map((stay, index) => {
              const remaining = calculateTimeRemaining(stay.first_seen, stay.time_limit_minutes || 120);
              const isOverstay = remaining < 0;
              
              return (
                <TouchableOpacity 
                  key={stay.id} 
                  style={[
                    styles.activeCard,
                    isOverstay && styles.activeCardOverstay,
                    index > 0 && styles.cardMargin
                  ]}
                  onPress={() => clearTimedStay(stay.id)}
                >
                  <View style={styles.activeHeader}>
                    <Text style={styles.activePlate}>{stay.plate_text}</Text>
                    {isOverstay && (
                      <View style={styles.overstayBadge}>
                        <Text style={styles.overstayBadgeText}>OVERSTAY</Text>
                      </View>
                    )}
                  </View>
                  
                  <View style={styles.activeTimeRow}>
                    <Text style={styles.activeTimeLabel}>
                      {isOverstay ? 'Over by:' : 'Time left:'}
                    </Text>
                    <Text style={[
                      styles.activeTimeValue,
                      isOverstay ? styles.timeOverstay : styles.timeOk
                    ]}>
                      {formatTimeRemaining(remaining)}
                    </Text>
                  </View>
                  
                  {stay.lot_name && (
                    <Text style={styles.activeLot}>{stay.lot_name}</Text>
                  )}
                  
                  <Text style={styles.tapToClear}>Tap to clear</Text>
                </TouchableOpacity>
              );
            })}
          </View>
        )}
      </View>

      {/* Section 2: Violations */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Violations ({violations.length})</Text>
        
        {violations.length === 0 ? (
          <Text style={styles.emptyText}>No violations</Text>
        ) : (
          <View>
            {violations.slice(0, 10).map((violation, index) => (
              <View key={violation.id} style={[
                styles.violationCard,
                index > 0 && styles.cardMargin
              ]}>
                <View style={styles.violationHeader}>
                  <Text style={styles.violationPlate}>{violation.plate_text}</Text>
                  <TouchableOpacity
                    style={styles.deleteButton}
                    onPress={() => deleteViolation(violation.id)}
                  >
                    <Text style={styles.deleteButtonText}>✕</Text>
                  </TouchableOpacity>
                </View>
                <View style={styles.violationDetails}>
                  <Text style={styles.violationReason}>
                    {violation.reason === 'exceeded_time' ? 'Time Exceeded' : 
                     violation.reason === 'no_permit' ? 'No Permit' : 
                     violation.reason.replace(/_/g, ' ')}
                  </Text>
                  <Text style={styles.violationTime}>{formatTimestamp(violation.timestamp)}</Text>
                </View>
              </View>
            ))}
          </View>
        )}
      </View>

      {/* Section 3: Recent Events */}
      <View style={styles.card}>
        <View style={styles.sectionHeader}>
          <Text style={styles.cardTitle}>Recent Scans ({events.length})</Text>
          {events.length > 5 && (
            <TouchableOpacity
              style={styles.expandButton}
              onPress={() => setShowAllEvents(!showAllEvents)}
            >
              <Text style={styles.expandButtonText}>
                {showAllEvents ? '▲ Less' : `▼ All`}
              </Text>
            </TouchableOpacity>
          )}
        </View>

        {displayedEvents.length === 0 ? (
          <Text style={styles.emptyText}>No scans yet</Text>
        ) : (
          <View>
            {displayedEvents.map((event, index) => (
              <TouchableOpacity
                key={event.id}
                style={[
                  styles.eventCard,
                  index > 0 && styles.cardMargin
                ]}
                onPress={() => handleEventClick(event)}
              >
                <View style={styles.eventContent}>
                  {eventImages[event.id] ? (
                    <Image
                      source={{ 
                        uri: eventImages[event.id].startsWith('data:') 
                          ? eventImages[event.id] 
                          : `data:image/jpeg;base64,${eventImages[event.id]}` 
                      }}
                      style={styles.eventThumbnail}
                      resizeMode="cover"
                    />
                  ) : (
                    <View style={styles.eventThumbnailEmpty}>
                      <Text style={styles.thumbnailEmptyText}>—</Text>
                    </View>
                  )}
                  
                  <View style={styles.eventInfo}>
                    <View style={styles.eventHeader}>
                      <View style={styles.eventMainInfo}>
                        <Text style={styles.eventPlate}>{event.plate_text}</Text>
                        {event.state && (
                          <Text style={styles.eventState}>{event.state}</Text>
                        )}
                      </View>
                      <View
                        style={[
                          styles.eventBadge,
                          event.result === 'approved' ? styles.approvedBadge : styles.violationBadge,
                        ]}
                      >
                        <Text style={styles.eventBadgeText}>
                          {event.result === 'approved' ? '✓' : '⚠'}
                        </Text>
                      </View>
                    </View>
                    <View style={styles.eventDetails}>
                      <Text style={styles.eventDetailText}>
                        {event.location} • {event.source || 'manual'}
                      </Text>
                      <Text style={styles.eventTime}>{formatTimestamp(event.timestamp)}</Text>
                    </View>
                    {event.confidence > 0 && (
                      <Text style={[styles.eventConfidence, { color: getConfidenceColor(event.confidence) }]}>
                        {(event.confidence * 100).toFixed(0)}%
                      </Text>
                    )}
                  </View>
                </View>
              </TouchableOpacity>
            ))}
          </View>
        )}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  header: {
    padding: 20,
    paddingTop: 60,
    backgroundColor: '#1a1a1a',
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#FFF',
    marginBottom: 4,
  },
  subtitle: {
    fontSize: 14,
    color: '#888',
  },
  errorCard: {
    backgroundColor: '#FF3B3020',
    margin: 12,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#FF3B30',
    alignItems: 'center',
  },
  errorText: {
    color: '#FF3B30',
    fontSize: 14,
    marginBottom: 12,
  },
  retryButton: {
    backgroundColor: '#FF3B30',
    paddingHorizontal: 20,
    paddingVertical: 8,
    borderRadius: 6,
  },
  retryButtonText: {
    color: '#FFF',
    fontSize: 14,
    fontWeight: '600',
  },
  card: {
    backgroundColor: '#1a1a1a',
    margin: 12,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#333',
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FFF',
    marginBottom: 12,
  },
  cardMargin: {
    marginTop: 10,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  expandButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: '#333',
    borderRadius: 6,
  },
  expandButtonText: {
    color: '#FFF',
    fontSize: 12,
    fontWeight: '600',
  },
  emptyText: {
    color: '#666',
    fontSize: 14,
    textAlign: 'center',
    paddingVertical: 12,
  },
  activeCard: {
    backgroundColor: '#000',
    padding: 14,
    borderRadius: 8,
    borderWidth: 2,
    borderColor: '#4CAF50',
  },
  activeCardOverstay: {
    borderColor: '#FF3B30',
  },
  activeHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  activePlate: {
    color: '#FFD700',
    fontSize: 20,
    fontWeight: 'bold',
    fontFamily: 'monospace',
    letterSpacing: 2,
  },
  overstayBadge: {
    backgroundColor: '#FF3B30',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  overstayBadgeText: {
    color: '#FFF',
    fontSize: 10,
    fontWeight: '700',
  },
  activeTimeRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  activeTimeLabel: {
    color: '#888',
    fontSize: 14,
  },
  activeTimeValue: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  timeOk: {
    color: '#4CAF50',
  },
  timeOverstay: {
    color: '#FF3B30',
  },
  activeLot: {
    color: '#888',
    fontSize: 12,
    marginTop: 8,
  },
  tapToClear: {
    color: '#007AFF',
    fontSize: 10,
    textAlign: 'right',
    marginTop: 8,
  },
  violationCard: {
    backgroundColor: '#000',
    padding: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#FF3B30',
  },
  violationHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  violationPlate: {
    color: '#FFD700',
    fontSize: 16,
    fontWeight: 'bold',
    fontFamily: 'monospace',
    letterSpacing: 1,
  },
  deleteButton: {
    backgroundColor: '#FF3B30',
    width: 24,
    height: 24,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  deleteButtonText: {
    color: '#FFF',
    fontSize: 14,
    fontWeight: 'bold',
  },
  violationDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  violationReason: {
    color: '#FF6B6B',
    fontSize: 12,
    fontWeight: '600',
  },
  violationTime: {
    color: '#666',
    fontSize: 11,
  },
  eventCard: {
    backgroundColor: '#000',
    padding: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#333',
  },
  eventContent: {
    flexDirection: 'row',
  },
  eventThumbnail: {
    width: 50,
    height: 50,
    borderRadius: 6,
    backgroundColor: '#333',
    marginRight: 12,
  },
  eventThumbnailEmpty: {
    width: 50,
    height: 50,
    borderRadius: 6,
    backgroundColor: '#222',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  thumbnailEmptyText: {
    color: '#444',
    fontSize: 16,
  },
  eventInfo: {
    flex: 1,
  },
  eventHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  eventMainInfo: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  eventPlate: {
    color: '#FFD700',
    fontSize: 14,
    fontWeight: 'bold',
    fontFamily: 'monospace',
    letterSpacing: 1,
    marginRight: 8,
  },
  eventState: {
    color: '#FFF',
    fontSize: 10,
    backgroundColor: '#007AFF',
    paddingHorizontal: 5,
    paddingVertical: 2,
    borderRadius: 3,
    fontWeight: '700',
    overflow: 'hidden',
  },
  eventBadge: {
    width: 24,
    height: 24,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  approvedBadge: {
    backgroundColor: '#4CAF50',
  },
  violationBadge: {
    backgroundColor: '#FF3B30',
  },
  eventBadgeText: {
    color: '#FFF',
    fontSize: 12,
    fontWeight: 'bold',
  },
  eventDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  eventDetailText: {
    color: '#888',
    fontSize: 10,
  },
  eventTime: {
    color: '#666',
    fontSize: 10,
  },
  eventConfidence: {
    fontSize: 10,
    fontWeight: '600',
    marginTop: 2,
  },
});