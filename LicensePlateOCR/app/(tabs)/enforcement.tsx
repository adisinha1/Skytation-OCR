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
  Pressable,
} from 'react-native';
import { useFocusEffect } from 'expo-router';
import { getZones, CampusZone } from '@/app/campusZones';

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
  time_limit_minutes?: number;
  lot_name?: string;
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
  time_limit_minutes: number;
  lot_name?: string;
}

interface Violation {
  id: number;
  plate_text: string;
  timestamp: string;
  location: string;
  reason: string;
}

const TIME_LIMIT_OPTIONS = [
  { label: '30 min', value: 30 },
  { label: '1 hr', value: 60 },
  { label: '2 hr', value: 120 },
  { label: '4 hr', value: 240 },
];

export default function EnforcementScreen() {
  const [events, setEvents] = useState<Event[]>([]);
  const [permits, setPermits] = useState<Permit[]>([]);
  const [timedStays, setTimedStays] = useState<TimedStay[]>([]);
  const [violations, setViolations] = useState<Violation[]>([]);
  const [zones, setZones] = useState<CampusZone[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [showAllEvents, setShowAllEvents] = useState(false);
  
  // Edit modal state
  const [editingEvent, setEditingEvent] = useState<Event | null>(null);
  const [showEditModal, setShowEditModal] = useState(false);
  const [editZoneType, setEditZoneType] = useState<'permit' | 'timed'>('timed');
  const [editTimeLimit, setEditTimeLimit] = useState(120);
  const [editLotName, setEditLotName] = useState('');
  const [editNotes, setEditNotes] = useState('');
  const [showTimeLimitPicker, setShowTimeLimitPicker] = useState(false);
  const [showLotPicker, setShowLotPicker] = useState(false);

  const loadData = async () => {
    try {
      const [eventsRes, permitsRes, timedRes, violationsRes, loadedZones] = await Promise.all([
        fetch(`${BACKEND_URL}/api/events`),
        fetch(`${BACKEND_URL}/api/permits`),
        fetch(`${BACKEND_URL}/api/timed_stays`),
        fetch(`${BACKEND_URL}/api/violations`),
        getZones(),
      ]);

      if (eventsRes.ok) setEvents(await eventsRes.json());
      if (permitsRes.ok) setPermits(await permitsRes.json());
      if (timedRes.ok) setTimedStays(await timedRes.json());
      if (violationsRes.ok) setViolations(await violationsRes.json());
      setZones(loadedZones);
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

  // Parse timestamp from backend - backend sends UTC timestamps
  // SQLAlchemy returns datetime as ISO string, we need to ensure it's parsed as UTC
  const parseTimestamp = (timestamp: string): Date => {
    if (!timestamp) return new Date();
    
    // If timestamp doesn't end with Z or have timezone offset, it's likely UTC from SQLAlchemy
    // SQLAlchemy with timezone=True sends format like "2024-01-15T17:30:00" (no Z but is UTC)
    let ts = timestamp;
    
    // Check if it has timezone info
    const hasTimezone = ts.endsWith('Z') || 
                        ts.includes('+') || 
                        (ts.includes('-') && ts.lastIndexOf('-') > 10);
    
    // If no timezone info, append Z to treat as UTC (since backend stores UTC)
    if (!hasTimezone && ts.includes('T')) {
      ts = ts + 'Z';
    }
    
    return new Date(ts);
  };

  const formatTimestamp = (timestamp: string) => {
    try {
      const date = parseTimestamp(timestamp);
      // toLocaleString will automatically convert UTC to local time
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
      const date = parseTimestamp(timestamp);
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
    const start = parseTimestamp(firstSeen);
    const now = new Date();
    const diffMin = (now.getTime() - start.getTime()) / 1000 / 60;
    return diffMin;
  };

  const formatDwellTime = (minutes: number) => {
    if (minutes < 0) {
      return '0s';
    } else if (minutes < 1) {
      return `${Math.floor(minutes * 60)}s`;
    } else if (minutes < 60) {
      return `${Math.floor(minutes)}m`;
    } else {
      const hours = Math.floor(minutes / 60);
      const mins = Math.floor(minutes % 60);
      return `${hours}h ${mins}m`;
    }
  };

  const calculateTimeRemaining = (firstSeen: string, limitMinutes: number = 120) => {
    const dwell = calculateDwell(firstSeen);
    const remaining = limitMinutes - dwell;
    return remaining;
  };

  // Helper function to get confidence color - threshold is now 85%
  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.85) return '#4CAF50'; // High - green
    if (confidence > 0.6) return '#FFA500';  // Medium - orange
    return '#F44336'; // Low - red
  };

  const handleEventClick = (event: Event) => {
    setEditingEvent(event);
    setEditZoneType(event.location as 'permit' | 'timed');
    setEditTimeLimit(event.time_limit_minutes || 120);
    setEditLotName(event.lot_name || '');
    setEditNotes(event.notes || '');
    setShowEditModal(true);
  };

  const saveEventChanges = async () => {
    if (!editingEvent) return;

    try {
      const response = await fetch(`${BACKEND_URL}/api/events/${editingEvent.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          location: editZoneType,
          time_limit_minutes: editTimeLimit,
          lot_name: editLotName,
          notes: editNotes,
        }),
      });

      if (response.ok) {
        Alert.alert('Success', 'Event updated successfully');
        setShowEditModal(false);
        await loadData();
      } else {
        Alert.alert('Error', 'Failed to update event');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to update event: ' + String(error));
    }
  };

  const deleteViolation = async (violationId: number) => {
    Alert.alert(
      'Delete Violation',
      'Are you sure you want to remove this violation from the database?',
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
                Alert.alert('Success', 'Violation deleted');
                await loadData();
              } else {
                Alert.alert('Error', 'Failed to delete violation');
              }
            } catch (error) {
              Alert.alert('Error', 'Failed to delete violation: ' + String(error));
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
                {event.confidence > 0 && (
                  <View style={styles.eventConfidence}>
                    <Text style={[styles.eventConfidenceText, { color: getConfidenceColor(event.confidence) }]}>
                      {(event.confidence * 100).toFixed(0)}% confidence
                    </Text>
                  </View>
                )}
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
              const timeLimit = stay.time_limit_minutes || 120;
              const remaining = calculateTimeRemaining(stay.first_seen, timeLimit);
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
                      <Text style={styles.timedStayLabel}>Time Limit:</Text>
                      <Text style={styles.timedStayValue}>{formatDwellTime(timeLimit)}</Text>
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
                    {stay.lot_name && (
                      <View style={styles.timedStayTimeRow}>
                        <Text style={styles.timedStayLabel}>Lot:</Text>
                        <Text style={styles.timedStayValue}>{stay.lot_name}</Text>
                      </View>
                    )}
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
                  <View style={styles.violationActions}>
                    <Text style={styles.violationReason}>
                      {violation.reason.replace(/_/g, ' ').toUpperCase()}
                    </Text>
                    <TouchableOpacity
                      style={styles.deleteButton}
                      onPress={() => deleteViolation(violation.id)}
                    >
                      <Text style={styles.deleteButtonText}>✕</Text>
                    </TouchableOpacity>
                  </View>
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
            <Text style={styles.modalTitle}>Edit Event</Text>
            
            {editingEvent && (
              <ScrollView style={styles.modalBody}>
                {/* Event Info */}
                <View style={styles.modalSection}>
                  <Text style={styles.modalSectionTitle}>Event Information</Text>
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
                    <Text style={styles.modalLabel}>Source:</Text>
                    <Text style={styles.modalValue}>{editingEvent.source || 'N/A'}</Text>
                  </View>
                  <View style={styles.modalRow}>
                    <Text style={styles.modalLabel}>Time:</Text>
                    <Text style={styles.modalValue}>{formatTimestamp(editingEvent.timestamp)}</Text>
                  </View>
                  <View style={styles.modalRow}>
                    <Text style={styles.modalLabel}>Confidence:</Text>
                    <Text style={[styles.modalValue, { color: getConfidenceColor(editingEvent.confidence) }]}>
                      {(editingEvent.confidence * 100).toFixed(1)}%
                    </Text>
                  </View>
                </View>

                {/* Zone Type Toggle */}
                <View style={styles.modalSection}>
                  <Text style={styles.modalSectionTitle}>Zone Type</Text>
                  <View style={styles.zoneTypeToggle}>
                    <TouchableOpacity
                      style={[
                        styles.zoneTypeButton,
                        editZoneType === 'permit' && styles.zoneTypeButtonActive
                      ]}
                      onPress={() => setEditZoneType('permit')}
                    >
                      <Text style={[
                        styles.zoneTypeButtonText,
                        editZoneType === 'permit' && styles.zoneTypeButtonTextActive
                      ]}>
                        Permit Zone
                      </Text>
                    </TouchableOpacity>
                    <TouchableOpacity
                      style={[
                        styles.zoneTypeButton,
                        editZoneType === 'timed' && styles.zoneTypeButtonActive
                      ]}
                      onPress={() => setEditZoneType('timed')}
                    >
                      <Text style={[
                        styles.zoneTypeButtonText,
                        editZoneType === 'timed' && styles.zoneTypeButtonTextActive
                      ]}>
                        Timed Zone
                      </Text>
                    </TouchableOpacity>
                  </View>
                </View>

                {/* Time Limit Selector (for timed zones) */}
                {editZoneType === 'timed' && (
                  <View style={styles.modalSection}>
                    <Text style={styles.modalSectionTitle}>Time Limit</Text>
                    <TouchableOpacity
                      style={styles.pickerButton}
                      onPress={() => setShowTimeLimitPicker(true)}
                    >
                      <Text style={styles.pickerButtonText}>
                        {TIME_LIMIT_OPTIONS.find(opt => opt.value === editTimeLimit)?.label || `${editTimeLimit} min`}
                      </Text>
                      <Text style={styles.pickerButtonArrow}>▼</Text>
                    </TouchableOpacity>
                  </View>
                )}

                {/* Lot Selector */}
                <View style={styles.modalSection}>
                  <Text style={styles.modalSectionTitle}>Parking Lot</Text>
                  <TouchableOpacity
                    style={styles.pickerButton}
                    onPress={() => setShowLotPicker(true)}
                  >
                    <Text style={styles.pickerButtonText}>
                      {editLotName || 'Select lot...'}
                    </Text>
                    <Text style={styles.pickerButtonArrow}>▼</Text>
                  </TouchableOpacity>
                </View>

                {/* Notes Field */}
                <View style={styles.modalSection}>
                  <Text style={styles.modalSectionTitle}>Notes</Text>
                  <TextInput
                    style={styles.notesInput}
                    value={editNotes}
                    onChangeText={setEditNotes}
                    placeholder="Add notes (e.g., Vehicle description, Warning issued)"
                    placeholderTextColor="#666"
                    multiline
                    numberOfLines={3}
                  />
                </View>
              </ScrollView>
            )}
            
            <View style={styles.modalButtons}>
              <TouchableOpacity
                style={styles.modalSaveButton}
                onPress={saveEventChanges}
              >
                <Text style={styles.buttonText}>Save Changes</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.modalCancelButton}
                onPress={() => setShowEditModal(false)}
              >
                <Text style={styles.buttonText}>Cancel</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      {/* Time Limit Picker Modal */}
      <Modal
        visible={showTimeLimitPicker}
        transparent={true}
        animationType="slide"
        onRequestClose={() => setShowTimeLimitPicker(false)}
      >
        <View style={styles.pickerModalOverlay}>
          <Pressable 
            style={styles.pickerModalDismiss}
            onPress={() => setShowTimeLimitPicker(false)}
          />
          <View style={styles.pickerModalContent}>
            <Text style={styles.pickerModalTitle}>Select Time Limit</Text>
            <View style={styles.pickerOptionsContainer}>
              {TIME_LIMIT_OPTIONS.map((option) => (
                <Pressable
                  key={option.value}
                  style={styles.pickerOption}
                  onPress={() => {
                    setEditTimeLimit(option.value);
                    setShowTimeLimitPicker(false);
                  }}
                >
                  <Text style={[
                    styles.pickerOptionText,
                    editTimeLimit === option.value && styles.pickerOptionTextSelected
                  ]}>
                    {option.label}
                  </Text>
                  {editTimeLimit === option.value && (
                    <Text style={styles.pickerCheckmark}>✓</Text>
                  )}
                </Pressable>
              ))}
            </View>
            <Pressable
              style={styles.pickerModalCloseButton}
              onPress={() => setShowTimeLimitPicker(false)}
            >
              <Text style={styles.buttonText}>Close</Text>
            </Pressable>
          </View>
        </View>
      </Modal>

      {/* Lot Picker Modal */}
      <Modal
        visible={showLotPicker}
        transparent={true}
        animationType="slide"
        onRequestClose={() => setShowLotPicker(false)}
      >
        <View style={styles.pickerModalOverlay}>
          <Pressable 
            style={styles.pickerModalDismiss}
            onPress={() => setShowLotPicker(false)}
          />
          <View style={styles.pickerModalContent}>
            <Text style={styles.pickerModalTitle}>Select Parking Lot</Text>
            <ScrollView style={styles.pickerScrollView}>
              <Pressable
                style={styles.pickerOption}
                onPress={() => {
                  setEditLotName('');
                  setShowLotPicker(false);
                }}
              >
                <Text style={[
                  styles.pickerOptionText,
                  editLotName === '' && styles.pickerOptionTextSelected
                ]}>
                  None
                </Text>
                {editLotName === '' && (
                  <Text style={styles.pickerCheckmark}>✓</Text>
                )}
              </Pressable>
              {zones.length > 0 ? (
                zones.map((zone) => (
                  <Pressable
                    key={zone.id}
                    style={styles.pickerOption}
                    onPress={() => {
                      setEditLotName(zone.name);
                      setShowLotPicker(false);
                    }}
                  >
                    <Text style={[
                      styles.pickerOptionText,
                      editLotName === zone.name && styles.pickerOptionTextSelected
                    ]}>
                      {zone.name} ({zone.code})
                    </Text>
                    {editLotName === zone.name && (
                      <Text style={styles.pickerCheckmark}>✓</Text>
                    )}
                  </Pressable>
                ))
              ) : (
                <View style={styles.pickerEmptyState}>
                  <Text style={styles.emptyText}>No lots configured.</Text>
                  <Text style={styles.emptyTextSmall}>Add them in the Explore tab.</Text>
                </View>
              )}
            </ScrollView>
            <Pressable
              style={styles.pickerModalCloseButton}
              onPress={() => setShowLotPicker(false)}
            >
              <Text style={styles.buttonText}>Close</Text>
            </Pressable>
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
  emptyTextSmall: {
    color: '#555',
    fontSize: 12,
    textAlign: 'center',
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
  eventConfidence: {
    marginTop: 4,
  },
  eventConfidenceText: {
    fontSize: 11,
    fontWeight: '600',
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
  violationActions: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  violationReason: {
    color: '#FF6B6B',
    fontSize: 12,
    fontWeight: '600',
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
    maxHeight: '90%',
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
    maxHeight: 500,
  },
  modalSection: {
    marginBottom: 20,
  },
  modalSectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#FFF',
    marginBottom: 12,
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
  zoneTypeToggle: {
    flexDirection: 'row',
    gap: 12,
  },
  zoneTypeButton: {
    flex: 1,
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#333',
    alignItems: 'center',
    backgroundColor: '#000',
  },
  zoneTypeButtonActive: {
    backgroundColor: '#007AFF',
    borderColor: '#007AFF',
  },
  zoneTypeButtonText: {
    color: '#888',
    fontSize: 14,
    fontWeight: '600',
  },
  zoneTypeButtonTextActive: {
    color: '#FFF',
  },
  pickerButton: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#000',
    borderWidth: 1,
    borderColor: '#333',
    borderRadius: 8,
    padding: 12,
  },
  pickerButtonText: {
    color: '#FFF',
    fontSize: 14,
  },
  pickerButtonArrow: {
    color: '#888',
    fontSize: 12,
  },
  notesInput: {
    backgroundColor: '#000',
    borderWidth: 1,
    borderColor: '#333',
    borderRadius: 8,
    padding: 12,
    color: '#FFF',
    fontSize: 14,
    minHeight: 80,
    textAlignVertical: 'top',
  },
  modalButtons: {
    gap: 12,
    marginTop: 20,
  },
  modalSaveButton: {
    backgroundColor: '#4CAF50',
    padding: 14,
    borderRadius: 8,
    alignItems: 'center',
  },
  modalCancelButton: {
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
  pickerModalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    justifyContent: 'flex-end',
  },
  pickerModalDismiss: {
    flex: 1,
  },
  pickerModalContent: {
    backgroundColor: '#1a1a1a',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    padding: 20,
    maxHeight: '70%',
  },
  pickerModalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFF',
    marginBottom: 16,
    textAlign: 'center',
  },
  pickerScrollView: {
    maxHeight: 300,
  },
  pickerOptionsContainer: {
    marginBottom: 10,
  },
  pickerOption: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  pickerOptionText: {
    color: '#FFF',
    fontSize: 16,
  },
  pickerOptionTextSelected: {
    color: '#007AFF',
    fontWeight: 'bold',
  },
  pickerCheckmark: {
    color: '#007AFF',
    fontSize: 18,
    fontWeight: 'bold',
  },
  pickerEmptyState: {
    padding: 20,
    alignItems: 'center',
  },
  pickerModalCloseButton: {
    backgroundColor: '#333',
    padding: 14,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 16,
  },
});