import React, { useState, useCallback, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Alert,
  RefreshControl,
  Modal,
} from 'react-native';
import { useFocusEffect } from 'expo-router';

const BACKEND_URL = 'http://10.0.0.67:8000'; // Update with your computer's IP

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
}

interface Permit {
  id: number;
  plate_text: string;
  state?: string;
  permit_type: string;
  notes?: string;
}

interface TimedStay {
  id: number;
  plate_text: string;
  first_seen: string;
  last_seen: string;
}

interface Violation {
  id: number;
  plate_text: string;
  timestamp: string;
  location: string;
  reason: string;
}

export default function EnforcementScreen() {
  const [events, setEvents] = useState<Event[]>([]);
  const [permits, setPermits] = useState<Permit[]>([]);
  const [timedStays, setTimedStays] = useState<TimedStay[]>([]);
  const [violations, setViolations] = useState<Violation[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [showAllEvents, setShowAllEvents] = useState(false);
  
  // Edit modal state
  const [editingEvent, setEditingEvent] = useState<Event | null>(null);
  const [showEditModal, setShowEditModal] = useState(false);

  const loadData = async () => {
    try {
      const [eventsRes, permitsRes, timedRes, violationsRes] = await Promise.all([
        fetch(`${BACKEND_URL}/api/events`),
        fetch(`${BACKEND_URL}/api/permits`),
        fetch(`${BACKEND_URL}/api/timed_stays`),
        fetch(`${BACKEND_URL}/api/violations`),
      ]);

      if (eventsRes.ok) setEvents(await eventsRes.json());
      if (permitsRes.ok) setPermits(await permitsRes.json());
      if (timedRes.ok) setTimedStays(await timedRes.json());
      if (violationsRes.ok) setViolations(await violationsRes.json());
    } catch (error) {
      console.error('Error loading data:', error);
    }
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadData();
    setRefreshing(false);
  };

  useFocusEffect(
    useCallback(() => {
      loadData();
    }, [])
  );

  // Update timer every second for countdown
  useEffect(() => {
    const interval = setInterval(() => {
      // Force re-render to update countdown timers
      setTimedStays(stays => [...stays]);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  const formatTimestamp = (timestamp: string) => {
    try {
      const date = new Date(timestamp);
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

  const formatTime = (timestamp: string) => {
    try {
      const date = new Date(timestamp);
      return date.toLocaleTimeString('en-US', {
        hour: 'numeric',
        minute: '2-digit',
        hour12: true,
      });
    } catch {
      return timestamp;
    }
  };

  const calculateDwell = (firstSeen: string) => {
    const start = new Date(firstSeen);
    const now = new Date();
    const diffMin = (now.getTime() - start.getTime()) / 1000 / 60;
    return diffMin;
  };

  const formatDwellTime = (minutes: number) => {
    if (minutes < 1) {
      return `${Math.floor(minutes * 60)}s`;
    } else if (minutes < 60) {
      return `${Math.floor(minutes)}m`;
    } else {
      const hours = Math.floor(minutes / 60);
      const mins = Math.floor(minutes % 60);
      return `${hours}h ${mins}m`;
    }
  };

  const calculateTimeRemaining = (firstSeen: string, limitMinutes: number = 2) => {
    const dwell = calculateDwell(firstSeen);
    const remaining = limitMinutes - dwell;
    return remaining;
  };

  const handleEventClick = (event: Event) => {
    setEditingEvent(event);
    setShowEditModal(true);
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

      {/* Section 1: Recent Events */}
      <View style={styles.card}>
        <View style={styles.sectionHeader}>
          <Text style={styles.cardTitle}>Recent Events ({events.length})</Text>
          {events.length > 5 && (
            <TouchableOpacity
              style={styles.expandButton}
              onPress={() => setShowAllEvents(!showAllEvents)}
            >
              <Text style={styles.expandButtonText}>
                {showAllEvents ? '▲ Show Less' : `▼ Show All (${events.length})`}
              </Text>
            </TouchableOpacity>
          )}
        </View>

        {displayedEvents.length === 0 ? (
          <Text style={styles.emptyText}>No events yet</Text>
        ) : (
          <View style={styles.eventsList}>
            {displayedEvents.map((event) => (
              <TouchableOpacity
                key={event.id}
                style={styles.eventCard}
                onPress={() => handleEventClick(event)}
              >
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
                    {event.location} • {event.source || 'unknown'}
                  </Text>
                  <Text style={styles.eventTime}>{formatTimestamp(event.timestamp)}</Text>
                </View>
                {event.notes && (
                  <Text style={styles.eventNotes}>{event.notes}</Text>
                )}
              </TouchableOpacity>
            ))}
          </View>
        )}
      </View>

      {/* Section 2: Active Parking (Timed Stays) */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Active Parking ({timedStays.length})</Text>
        
        {timedStays.length === 0 ? (
          <Text style={styles.emptyText}>No active parked vehicles</Text>
        ) : (
          <View style={styles.timedStaysList}>
            {timedStays.map((stay) => {
              const dwellMinutes = calculateDwell(stay.first_seen);
              const remaining = calculateTimeRemaining(stay.first_seen);
              const isOverstay = remaining < 0;
              
              return (
                <View key={stay.id} style={[
                  styles.timedStayCard,
                  isOverstay && styles.timedStayCardOverstay
                ]}>
                  <View style={styles.timedStayHeader}>
                    <Text style={styles.timedStayPlate}>{stay.plate_text}</Text>
                    {isOverstay && (
                      <Text style={styles.overstayBadge}>OVERSTAY</Text>
                    )}
                  </View>
                  
                  <View style={styles.timedStayTimes}>
                    <View style={styles.timedStayTimeRow}>
                      <Text style={styles.timedStayLabel}>Scanned In:</Text>
                      <Text style={styles.timedStayValue}>{formatTime(stay.first_seen)}</Text>
                    </View>
                    <View style={styles.timedStayTimeRow}>
                      <Text style={styles.timedStayLabel}>Time Parked:</Text>
                      <Text style={styles.timedStayValue}>{formatDwellTime(dwellMinutes)}</Text>
                    </View>
                    <View style={styles.timedStayTimeRow}>
                      <Text style={styles.timedStayLabel}>Time Remaining:</Text>
                      <Text style={[
                        styles.timedStayValue,
                        isOverstay ? styles.timedStayOverstay : styles.timedStayOk
                      ]}>
                        {isOverstay 
                          ? `+${formatDwellTime(-remaining)}` 
                          : formatDwellTime(remaining)
                        }
                      </Text>
                    </View>
                  </View>
                </View>
              );
            })}
          </View>
        )}
      </View>

      {/* Section 3: Violations */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Recent Violations ({violations.length})</Text>
        
        {violations.length === 0 ? (
          <Text style={styles.emptyText}>No violations</Text>
        ) : (
          <View style={styles.violationsList}>
            {violations.slice(0, 10).map((violation) => (
              <View key={violation.id} style={styles.violationCard}>
                <View style={styles.violationHeader}>
                  <Text style={styles.violationPlate}>{violation.plate_text}</Text>
                  <Text style={styles.violationReason}>
                    {violation.reason.replace(/_/g, ' ').toUpperCase()}
                  </Text>
                </View>
                <View style={styles.violationDetails}>
                  <Text style={styles.violationLocation}>{violation.location}</Text>
                  <Text style={styles.violationTime}>{formatTimestamp(violation.timestamp)}</Text>
                </View>
              </View>
            ))}
          </View>
        )}
      </View>

      {/* Edit Event Modal */}
      <Modal
        visible={showEditModal}
        transparent={true}
        animationType="slide"
        onRequestClose={() => setShowEditModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Event Details</Text>
            
            {editingEvent && (
              <View style={styles.modalBody}>
                <View style={styles.modalRow}>
                  <Text style={styles.modalLabel}>Plate:</Text>
                  <Text style={styles.modalValue}>{editingEvent.plate_text}</Text>
                </View>
                {editingEvent.state && (
                  <View style={styles.modalRow}>
                    <Text style={styles.modalLabel}>State:</Text>
                    <Text style={styles.modalValue}>{editingEvent.state}</Text>
                  </View>
                )}
                <View style={styles.modalRow}>
                  <Text style={styles.modalLabel}>Zone:</Text>
                  <Text style={styles.modalValue}>{editingEvent.location}</Text>
                </View>
                <View style={styles.modalRow}>
                  <Text style={styles.modalLabel}>Source:</Text>
                  <Text style={styles.modalValue}>{editingEvent.source || 'N/A'}</Text>
                </View>
                <View style={styles.modalRow}>
                  <Text style={styles.modalLabel}>Confidence:</Text>
                  <Text style={styles.modalValue}>{(editingEvent.confidence * 100).toFixed(0)}%</Text>
                </View>
                <View style={styles.modalRow}>
                  <Text style={styles.modalLabel}>Result:</Text>
                  <Text style={[
                    styles.modalValue,
                    editingEvent.result === 'approved' ? styles.approvedText : styles.violationText
                  ]}>
                    {editingEvent.result.toUpperCase()}
                  </Text>
                </View>
                {editingEvent.notes && (
                  <View style={styles.modalRow}>
                    <Text style={styles.modalLabel}>Notes:</Text>
                    <Text style={styles.modalValue}>{editingEvent.notes}</Text>
                  </View>
                )}
                <View style={styles.modalRow}>
                  <Text style={styles.modalLabel}>Time:</Text>
                  <Text style={styles.modalValue}>{formatTimestamp(editingEvent.timestamp)}</Text>
                </View>
              </View>
            )}
            
            <View style={styles.modalButtons}>
              <TouchableOpacity
                style={styles.modalCloseButton}
                onPress={() => setShowEditModal(false)}
              >
                <Text style={styles.buttonText}>Close</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
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
  card: {
    backgroundColor: '#1a1a1a',
    margin: 12,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#333',
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FFF',
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
    paddingVertical: 20,
  },
  eventsList: {
    gap: 8,
  },
  eventCard: {
    backgroundColor: '#000',
    padding: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#333',
  },
  eventHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  eventMainInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  eventPlate: {
    color: '#FFD700',
    fontSize: 18,
    fontWeight: 'bold',
    fontFamily: 'monospace',
    letterSpacing: 1,
  },
  eventState: {
    color: '#FFF',
    fontSize: 14,
    backgroundColor: '#007AFF',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
    fontWeight: '700',
  },
  eventBadge: {
    width: 32,
    height: 32,
    borderRadius: 16,
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
    fontSize: 16,
    fontWeight: 'bold',
  },
  eventDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  eventDetailText: {
    color: '#888',
    fontSize: 12,
  },
  eventTime: {
    color: '#666',
    fontSize: 11,
  },
  eventNotes: {
    color: '#AAA',
    fontSize: 11,
    fontStyle: 'italic',
    marginTop: 4,
  },
  timedStaysList: {
    gap: 12,
  },
  timedStayCard: {
    backgroundColor: '#000',
    padding: 12,
    borderRadius: 8,
    borderWidth: 2,
    borderColor: '#4CAF50',
  },
  timedStayCardOverstay: {
    borderColor: '#FF3B30',
  },
  timedStayHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  timedStayPlate: {
    color: '#FFD700',
    fontSize: 18,
    fontWeight: 'bold',
    fontFamily: 'monospace',
    letterSpacing: 1,
  },
  overstayBadge: {
    color: '#FFF',
    fontSize: 11,
    backgroundColor: '#FF3B30',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
    fontWeight: '700',
  },
  timedStayTimes: {
    gap: 6,
  },
  timedStayTimeRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  timedStayLabel: {
    color: '#888',
    fontSize: 13,
  },
  timedStayValue: {
    color: '#FFF',
    fontSize: 13,
    fontWeight: '600',
  },
  timedStayOk: {
    color: '#4CAF50',
  },
  timedStayOverstay: {
    color: '#FF3B30',
    fontWeight: 'bold',
  },
  violationsList: {
    gap: 8,
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
  violationReason: {
    color: '#FF6B6B',
    fontSize: 12,
    fontWeight: '600',
  },
  violationDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  violationLocation: {
    color: '#888',
    fontSize: 12,
  },
  violationTime: {
    color: '#666',
    fontSize: 11,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.9)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  modalContent: {
    backgroundColor: '#1a1a1a',
    borderRadius: 16,
    padding: 24,
    width: '100%',
    maxWidth: 400,
    borderWidth: 1,
    borderColor: '#333',
  },
  modalTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#FFF',
    marginBottom: 20,
    textAlign: 'center',
  },
  modalBody: {
    gap: 12,
  },
  modalRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  modalLabel: {
    color: '#888',
    fontSize: 14,
    fontWeight: '600',
  },
  modalValue: {
    color: '#FFF',
    fontSize: 14,
    flex: 1,
    textAlign: 'right',
  },
  approvedText: {
    color: '#4CAF50',
    fontWeight: 'bold',
  },
  violationText: {
    color: '#FF3B30',
    fontWeight: 'bold',
  },
  modalButtons: {
    marginTop: 20,
  },
  modalCloseButton: {
    backgroundColor: '#333',
    padding: 14,
    borderRadius: 8,
    alignItems: 'center',
  },
  buttonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: '600',
  },
});
