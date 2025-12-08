import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Alert,
  Image,
  Modal,
  ActivityIndicator,
} from 'react-native';
import { useLocalSearchParams, useRouter, Stack } from 'expo-router';

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
  image_data?: string;
}

interface Zone {
  id: number;
  name: string;
  code: string;
  zone_type: string;
  default_time_limit: number;
}

interface TimeLimitOption {
  label: string;
  value: number;
}

const TIME_LIMIT_OPTIONS: TimeLimitOption[] = [
  { label: '30 min', value: 30 },
  { label: '1 hr', value: 60 },
  { label: '2 hr', value: 120 },
  { label: '4 hr', value: 240 },
];

export default function EditEventScreen() {
  const router = useRouter();
  const { eventId } = useLocalSearchParams<{ eventId: string }>();
  
  const [event, setEvent] = useState<Event | null>(null);
  const [zones, setZones] = useState<Zone[]>([]);
  const [loading, setLoading] = useState(true);
  
  // Edit state
  const [zoneType, setZoneType] = useState<'permit' | 'timed'>('timed');
  const [timeLimit, setTimeLimit] = useState(120);
  const [lotName, setLotName] = useState('');
  const [notes, setNotes] = useState('');
  
  // Picker states
  const [timeLimitPickerVisible, setTimeLimitPickerVisible] = useState(false);
  const [lotPickerVisible, setLotPickerVisible] = useState(false);

  useEffect(() => {
    loadData();
  }, [eventId]);

  const loadData = async () => {
    try {
      setLoading(true);
      
      const [eventRes, zonesRes] = await Promise.all([
        fetch(`${BACKEND_URL}/api/events/${eventId}`),
        fetch(`${BACKEND_URL}/api/zones`),
      ]);

      if (eventRes.ok) {
        const eventData = await eventRes.json();
        setEvent(eventData);
        setZoneType(eventData.location as 'permit' | 'timed');
        setTimeLimit(eventData.time_limit_minutes || 120);
        setLotName(eventData.lot_name || '');
        setNotes(eventData.notes || '');
      } else {
        Alert.alert('Error', 'Event not found');
        router.back();
      }

      if (zonesRes.ok) {
        setZones(await zonesRes.json());
      }
    } catch (error) {
      console.error('Error loading data:', error);
      Alert.alert('Error', 'Failed to load event');
    } finally {
      setLoading(false);
    }
  };

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
        weekday: 'short',
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
        hour12: true,
      });
    } catch {
      return timestamp;
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.85) return '#4CAF50';
    if (confidence > 0.6) return '#FFA500';
    return '#F44336';
  };

  const getTimeLimitLabel = (value: number) => {
    const option = TIME_LIMIT_OPTIONS.find(opt => opt.value === value);
    return option?.label || `${value} min`;
  };

  const handleSave = async () => {
    if (!event) return;

    try {
      const response = await fetch(`${BACKEND_URL}/api/events/${event.id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          location: zoneType,
          time_limit_minutes: timeLimit,
          lot_name: lotName,
          notes: notes,
        }),
      });

      if (response.ok) {
        Alert.alert('Success', 'Event updated successfully', [
          { text: 'OK', onPress: () => router.back() }
        ]);
      } else {
        Alert.alert('Error', 'Failed to update event');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to update event: ' + String(error));
    }
  };

  const handleDelete = async () => {
    if (!event) return;

    Alert.alert(
      'Delete Event',
      `Are you sure you want to delete this scan for ${event.plate_text}? This action cannot be undone.`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              const response = await fetch(`${BACKEND_URL}/api/events/${event.id}`, {
                method: 'DELETE',
              });

              if (response.ok) {
                Alert.alert('Deleted', 'Event has been deleted', [
                  { text: 'OK', onPress: () => router.back() }
                ]);
              } else {
                Alert.alert('Error', 'Failed to delete event');
              }
            } catch (error) {
              Alert.alert('Error', 'Failed to delete event: ' + String(error));
            }
          },
        },
      ]
    );
  };

  if (loading) {
    return (
      <View style={styles.loadingContainer}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={styles.loadingText}>Loading event...</Text>
      </View>
    );
  }

  if (!event) {
    return (
      <View style={styles.errorContainer}>
        <Text style={styles.errorText}>Event not found</Text>
        <TouchableOpacity style={styles.backButton} onPress={() => router.back()}>
          <Text style={styles.backButtonText}>Go Back</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Hide the default Expo Router header */}
      <Stack.Screen options={{ headerShown: false }} />
      
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity style={styles.headerBackButton} onPress={() => router.back()}>
          <Text style={styles.headerBackText}>← Back</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Edit Event</Text>
        <View style={styles.headerSpacer} />
      </View>

      <ScrollView style={styles.content}>
        {/* Image Section */}
        {event.image_data ? (
          <View style={styles.imageSection}>
            <Image
              source={{ uri: event.image_data.startsWith('data:') ? event.image_data : `data:image/jpeg;base64,${event.image_data}` }}
              style={styles.eventImage}
              resizeMode="contain"
            />
          </View>
        ) : (
          <View style={styles.noImageSection}>
            <Text style={styles.noImageText}>No image available</Text>
          </View>
        )}

        {/* Plate Info */}
        <View style={styles.plateSection}>
          <Text style={styles.plateText}>{event.plate_text}</Text>
          {event.state && (
            <Text style={styles.stateText}>{event.state}</Text>
          )}
        </View>

        {/* Event Details */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Scan Details</Text>
          <View style={styles.detailRow}>
            <Text style={styles.detailLabel}>Time:</Text>
            <Text style={styles.detailValue}>{formatTimestamp(event.timestamp)}</Text>
          </View>
          <View style={styles.detailRow}>
            <Text style={styles.detailLabel}>Source:</Text>
            <Text style={styles.detailValue}>{event.source || 'N/A'}</Text>
          </View>
          <View style={styles.detailRow}>
            <Text style={styles.detailLabel}>Result:</Text>
            <Text style={[
              styles.detailValue,
              event.result === 'approved' ? styles.approvedText : styles.violationText
            ]}>
              {event.result.toUpperCase()}
            </Text>
          </View>
          <View style={styles.detailRow}>
            <Text style={styles.detailLabel}>Confidence:</Text>
            <Text style={[styles.detailValue, { color: getConfidenceColor(event.confidence) }]}>
              {(event.confidence * 100).toFixed(1)}%
            </Text>
          </View>
        </View>

        {/* Edit Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Edit Classification</Text>
          
          {/* Zone Type Toggle */}
          <View style={styles.fieldGroup}>
            <Text style={styles.fieldLabel}>Zone Type</Text>
            <View style={styles.zoneTypeToggle}>
              <TouchableOpacity
                style={[
                  styles.zoneTypeButton,
                  zoneType === 'permit' && styles.zoneTypeButtonActive
                ]}
                onPress={() => setZoneType('permit')}
              >
                <Text style={[
                  styles.zoneTypeButtonText,
                  zoneType === 'permit' && styles.zoneTypeButtonTextActive
                ]}>
                  Permit
                </Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[
                  styles.zoneTypeButton,
                  zoneType === 'timed' && styles.zoneTypeButtonActive
                ]}
                onPress={() => setZoneType('timed')}
              >
                <Text style={[
                  styles.zoneTypeButtonText,
                  zoneType === 'timed' && styles.zoneTypeButtonTextActive
                ]}>
                  Timed
                </Text>
              </TouchableOpacity>
            </View>
          </View>

          {/* Time Limit */}
          {zoneType === 'timed' && (
            <View style={styles.fieldGroup}>
              <Text style={styles.fieldLabel}>Time Limit</Text>
              <TouchableOpacity
                style={styles.dropdownButton}
                onPress={() => setTimeLimitPickerVisible(true)}
              >
                <Text style={styles.dropdownButtonText}>
                  {getTimeLimitLabel(timeLimit)}
                </Text>
                <Text style={styles.dropdownArrow}>▼</Text>
              </TouchableOpacity>
            </View>
          )}

          {/* Parking Lot */}
          <View style={styles.fieldGroup}>
            <Text style={styles.fieldLabel}>Parking Lot</Text>
            <TouchableOpacity
              style={styles.dropdownButton}
              onPress={() => setLotPickerVisible(true)}
            >
              <Text style={styles.dropdownButtonText}>
                {lotName || 'Select lot...'}
              </Text>
              <Text style={styles.dropdownArrow}>▼</Text>
            </TouchableOpacity>
          </View>

          {/* Notes */}
          <View style={styles.fieldGroup}>
            <Text style={styles.fieldLabel}>Notes</Text>
            <TextInput
              style={styles.notesInput}
              value={notes}
              onChangeText={setNotes}
              placeholder="Add notes about this scan..."
              placeholderTextColor="#666"
              multiline
              numberOfLines={4}
            />
          </View>
        </View>

        {/* Action Buttons */}
        <View style={styles.actionButtons}>
          <TouchableOpacity style={styles.saveButton} onPress={handleSave}>
            <Text style={styles.saveButtonText}>Save Changes</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.cancelButton} onPress={() => router.back()}>
            <Text style={styles.cancelButtonText}>Cancel</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.deleteEventButton} onPress={handleDelete}>
            <Text style={styles.deleteEventButtonText}>Delete This Scan</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>

      {/* Time Limit Picker Modal */}
      <Modal
        visible={timeLimitPickerVisible}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setTimeLimitPickerVisible(false)}
      >
        <TouchableOpacity 
          style={styles.pickerOverlay}
          activeOpacity={1}
          onPress={() => setTimeLimitPickerVisible(false)}
        >
          <View style={styles.pickerContainer}>
            <Text style={styles.pickerTitle}>Select Time Limit</Text>
            {TIME_LIMIT_OPTIONS.map((option) => (
              <TouchableOpacity
                key={option.value}
                style={styles.pickerItem}
                onPress={() => {
                  setTimeLimit(option.value);
                  setTimeLimitPickerVisible(false);
                }}
              >
                <Text style={[
                  styles.pickerItemText,
                  timeLimit === option.value && styles.pickerItemTextSelected
                ]}>
                  {option.label}
                </Text>
                {timeLimit === option.value && (
                  <Text style={styles.pickerCheckmark}>✓</Text>
                )}
              </TouchableOpacity>
            ))}
          </View>
        </TouchableOpacity>
      </Modal>

      {/* Lot Picker Modal */}
      <Modal
        visible={lotPickerVisible}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setLotPickerVisible(false)}
      >
        <TouchableOpacity 
          style={styles.pickerOverlay}
          activeOpacity={1}
          onPress={() => setLotPickerVisible(false)}
        >
          <View style={styles.pickerContainer}>
            <Text style={styles.pickerTitle}>Select Parking Lot</Text>
            <ScrollView style={styles.pickerScrollView}>
              <TouchableOpacity
                style={styles.pickerItem}
                onPress={() => {
                  setLotName('');
                  setLotPickerVisible(false);
                }}
              >
                <Text style={[
                  styles.pickerItemText,
                  lotName === '' && styles.pickerItemTextSelected
                ]}>
                  None
                </Text>
                {lotName === '' && (
                  <Text style={styles.pickerCheckmark}>✓</Text>
                )}
              </TouchableOpacity>
              {zones.map((zone) => (
                <TouchableOpacity
                  key={zone.id}
                  style={styles.pickerItem}
                  onPress={() => {
                    setLotName(zone.name);
                    setLotPickerVisible(false);
                  }}
                >
                  <Text style={[
                    styles.pickerItemText,
                    lotName === zone.name && styles.pickerItemTextSelected
                  ]}>
                    {zone.name} ({zone.code})
                  </Text>
                  {lotName === zone.name && (
                    <Text style={styles.pickerCheckmark}>✓</Text>
                  )}
                </TouchableOpacity>
              ))}
            </ScrollView>
          </View>
        </TouchableOpacity>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  loadingContainer: {
    flex: 1,
    backgroundColor: '#000',
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: '#888',
    marginTop: 12,
  },
  errorContainer: {
    flex: 1,
    backgroundColor: '#000',
    justifyContent: 'center',
    alignItems: 'center',
  },
  errorText: {
    color: '#FF3B30',
    fontSize: 18,
    marginBottom: 20,
  },
  backButton: {
    backgroundColor: '#333',
    padding: 12,
    borderRadius: 8,
  },
  backButtonText: {
    color: '#FFF',
    fontSize: 14,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: 60,
    paddingHorizontal: 16,
    paddingBottom: 16,
    backgroundColor: '#1a1a1a',
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  headerBackButton: {
    padding: 8,
  },
  headerBackText: {
    color: '#007AFF',
    fontSize: 16,
  },
  headerTitle: {
    color: '#FFF',
    fontSize: 18,
    fontWeight: 'bold',
  },
  headerSpacer: {
    width: 60,
  },
  content: {
    flex: 1,
  },
  imageSection: {
    backgroundColor: '#1a1a1a',
    padding: 16,
    alignItems: 'center',
  },
  eventImage: {
    width: '100%',
    height: 200,
    borderRadius: 8,
    backgroundColor: '#000',
  },
  noImageSection: {
    backgroundColor: '#1a1a1a',
    padding: 40,
    alignItems: 'center',
  },
  noImageText: {
    color: '#666',
    fontSize: 14,
  },
  plateSection: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    backgroundColor: '#1a1a1a',
    gap: 12,
  },
  plateText: {
    color: '#FFD700',
    fontSize: 32,
    fontWeight: 'bold',
    fontFamily: 'monospace',
    letterSpacing: 2,
  },
  stateText: {
    color: '#FFF',
    fontSize: 16,
    backgroundColor: '#007AFF',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 4,
    fontWeight: '700',
  },
  section: {
    backgroundColor: '#1a1a1a',
    margin: 12,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#333',
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#FFF',
    marginBottom: 16,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  detailLabel: {
    color: '#888',
    fontSize: 14,
  },
  detailValue: {
    color: '#FFF',
    fontSize: 14,
    fontWeight: '500',
  },
  approvedText: {
    color: '#4CAF50',
  },
  violationText: {
    color: '#FF3B30',
  },
  fieldGroup: {
    marginBottom: 16,
  },
  fieldLabel: {
    color: '#888',
    fontSize: 12,
    fontWeight: '600',
    marginBottom: 8,
  },
  zoneTypeToggle: {
    flexDirection: 'row',
    gap: 12,
  },
  zoneTypeButton: {
    flex: 1,
    paddingVertical: 12,
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
  dropdownButton: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#000',
    borderWidth: 1,
    borderColor: '#333',
    borderRadius: 8,
    padding: 12,
  },
  dropdownButtonText: {
    color: '#FFF',
    fontSize: 14,
  },
  dropdownArrow: {
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
    minHeight: 100,
    textAlignVertical: 'top',
  },
  actionButtons: {
    padding: 16,
    gap: 12,
  },
  saveButton: {
    backgroundColor: '#4CAF50',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  saveButtonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: '600',
  },
  cancelButton: {
    backgroundColor: '#333',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  cancelButtonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: '600',
  },
  deleteEventButton: {
    backgroundColor: '#FF3B30',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 8,
  },
  deleteEventButtonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: '600',
  },
  // Picker styles
  pickerOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  pickerContainer: {
    backgroundColor: '#1a1a1a',
    borderRadius: 16,
    padding: 20,
    width: '100%',
    maxWidth: 300,
    borderWidth: 1,
    borderColor: '#444',
  },
  pickerTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FFF',
    marginBottom: 16,
    textAlign: 'center',
  },
  pickerScrollView: {
    maxHeight: 250,
  },
  pickerItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 14,
    paddingHorizontal: 8,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  pickerItemText: {
    color: '#FFF',
    fontSize: 16,
    flex: 1,
  },
  pickerItemTextSelected: {
    color: '#007AFF',
    fontWeight: 'bold',
  },
  pickerCheckmark: {
    color: '#007AFF',
    fontSize: 18,
    fontWeight: 'bold',
    marginLeft: 8,
  },
});