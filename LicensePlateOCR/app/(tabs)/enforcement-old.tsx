import React, { useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  TextInput,
  Alert,
  RefreshControl,
} from 'react-native';
import { useFocusEffect } from 'expo-router';

const BACKEND_URL = 'http://10.0.0.67:8000'; // Update with your computer's IP

interface Event {
  id: number;
  plate_text: string;
  confidence: number;
  timestamp: string;
  location: string;
  result: string;
  notes?: string;
}

interface Permit {
  id: number;
  plate_text: string;
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
  
  // Form state
  const [plateInput, setPlateInput] = useState('');
  const [confidenceInput, setConfidenceInput] = useState('0.99');
  const [selectedLocation, setSelectedLocation] = useState<'permit' | 'timed'>('permit');
  const [newPermitPlate, setNewPermitPlate] = useState('');

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

  const submitOCREvent = async () => {
    if (!plateInput.trim()) {
      Alert.alert('Error', 'Please enter a plate number');
      return;
    }

    try {
      const response = await fetch(`${BACKEND_URL}/api/ocr_event`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          plate_text: plateInput.toUpperCase(),
          confidence: parseFloat(confidenceInput),
          timestamp: new Date().toISOString(),
          location: selectedLocation,
        }),
      });

      const result = await response.json();
      
      if (result.result === 'approved') {
        Alert.alert('✅ Approved', result.message);
      } else {
        Alert.alert('⛔ Violation', result.message);
      }

      await loadData();
      setPlateInput('');
    } catch (error) {
      Alert.alert('Error', 'Failed to submit event: ' + String(error));
    }
  };

  const addPermit = async () => {
    if (!newPermitPlate.trim()) {
      Alert.alert('Error', 'Please enter a plate number');
      return;
    }

    try {
      const response = await fetch(`${BACKEND_URL}/api/permits`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          plate_text: newPermitPlate.toUpperCase(),
          permit_type: 'A',
        }),
      });

      if (response.ok) {
        Alert.alert('Success', 'Permit added');
        setNewPermitPlate('');
        await loadData();
      } else {
        const error = await response.json();
        Alert.alert('Error', error.detail || 'Failed to add permit');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to add permit: ' + String(error));
    }
  };

  const seedPermits = async () => {
    try {
      await fetch(`${BACKEND_URL}/api/permits/seed`, { method: 'POST' });
      Alert.alert('Success', 'Sample permits seeded');
      await loadData();
    } catch (error) {
      Alert.alert('Error', 'Failed to seed permits');
    }
  };

  const resetTimed = async () => {
    Alert.alert(
      'Reset Timed Stays',
      'Are you sure you want to reset all timed parking stays?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Reset',
          style: 'destructive',
          onPress: async () => {
            try {
              await fetch(`${BACKEND_URL}/api/timed/reset`, { method: 'POST' });
              Alert.alert('Success', 'Timed stays reset');
              await loadData();
            } catch (error) {
              Alert.alert('Error', 'Failed to reset timed stays');
            }
          },
        },
      ]
    );
  };

  const formatTimestamp = (timestamp: string) => {
    try {
      return new Date(timestamp).toLocaleString();
    } catch {
      return timestamp;
    }
  };

  const calculateDwell = (firstSeen: string) => {
    const start = new Date(firstSeen);
    const now = new Date();
    const diffMin = (now.getTime() - start.getTime()) / 1000 / 60;
    return diffMin.toFixed(1);
  };

  return (
    <ScrollView
      style={styles.container}
      refreshControl={
        <RefreshControl refreshing={refreshing} onRefresh={onRefresh} />
      }
    >
      <View style={styles.header}>
        <Text style={styles.title}>Parking Enforcement</Text>
        <Text style={styles.subtitle}>Submit OCR events and manage permits</Text>
      </View>

      {/* Submit OCR Event */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Submit OCR Event</Text>
        
        <TextInput
          style={styles.input}
          placeholder="Plate Number (e.g., ABC123)"
          placeholderTextColor="#666"
          value={plateInput}
          onChangeText={setPlateInput}
          autoCapitalize="characters"
        />

        <TextInput
          style={styles.input}
          placeholder="Confidence (0.0 - 1.0)"
          placeholderTextColor="#666"
          value={confidenceInput}
          onChangeText={setConfidenceInput}
          keyboardType="decimal-pad"
        />

        <View style={styles.radioGroup}>
          <Text style={styles.label}>Location:</Text>
          <View style={styles.radioButtons}>
            <TouchableOpacity
              style={[styles.radioButton, selectedLocation === 'permit' && styles.radioButtonActive]}
              onPress={() => setSelectedLocation('permit')}
            >
              <Text style={[styles.radioText, selectedLocation === 'permit' && styles.radioTextActive]}>
                Permit Zone
              </Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.radioButton, selectedLocation === 'timed' && styles.radioButtonActive]}
              onPress={() => setSelectedLocation('timed')}
            >
              <Text style={[styles.radioText, selectedLocation === 'timed' && styles.radioTextActive]}>
                Timed Zone
              </Text>
            </TouchableOpacity>
          </View>
        </View>

        <TouchableOpacity style={styles.primaryButton} onPress={submitOCREvent}>
          <Text style={styles.buttonText}>Submit Event</Text>
        </TouchableOpacity>
      </View>

      {/* Manage Permits */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Manage Permits ({permits.length})</Text>
        
        <View style={styles.inputRow}>
          <TextInput
            style={[styles.input, { flex: 1 }]}
            placeholder="New Permit Plate"
            placeholderTextColor="#666"
            value={newPermitPlate}
            onChangeText={setNewPermitPlate}
            autoCapitalize="characters"
          />
          <TouchableOpacity style={styles.addButton} onPress={addPermit}>
            <Text style={styles.buttonText}>Add</Text>
          </TouchableOpacity>
        </View>

        <TouchableOpacity style={styles.secondaryButton} onPress={seedPermits}>
          <Text style={styles.buttonText}>Seed Sample Permits</Text>
        </TouchableOpacity>

        <ScrollView style={styles.listContainer} horizontal>
          {permits.map((permit) => (
            <View key={permit.id} style={styles.permitCard}>
              <Text style={styles.permitPlate}>{permit.plate_text}</Text>
              <Text style={styles.permitType}>Type: {permit.permit_type}</Text>
            </View>
          ))}
          {permits.length === 0 && (
            <Text style={styles.emptyText}>No permits yet</Text>
          )}
        </ScrollView>
      </View>

      {/* Timed Stays */}
      <View style={styles.card}>
        <View style={styles.cardHeader}>
          <Text style={styles.cardTitle}>Timed Stays ({timedStays.length})</Text>
          <TouchableOpacity style={styles.resetButton} onPress={resetTimed}>
            <Text style={styles.resetButtonText}>Reset All</Text>
          </TouchableOpacity>
        </View>

        {timedStays.map((stay) => (
          <View key={stay.id} style={styles.stayCard}>
            <Text style={styles.stayPlate}>{stay.plate_text}</Text>
            <Text style={styles.stayTime}>Dwell: {calculateDwell(stay.first_seen)} min</Text>
            <Text style={styles.stayTimestamp}>{formatTimestamp(stay.first_seen)}</Text>
          </View>
        ))}
        {timedStays.length === 0 && (
          <Text style={styles.emptyText}>No active timed stays</Text>
        )}
      </View>

      {/* Violations */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Recent Violations ({violations.length})</Text>
        {violations.slice(0, 10).map((violation) => (
          <View key={violation.id} style={styles.violationCard}>
            <Text style={styles.violationPlate}>{violation.plate_text}</Text>
            <Text style={styles.violationReason}>{violation.reason.replace('_', ' ')}</Text>
            <Text style={styles.violationLocation}>{violation.location}</Text>
            <Text style={styles.violationTime}>{formatTimestamp(violation.timestamp)}</Text>
          </View>
        ))}
        {violations.length === 0 && (
          <Text style={styles.emptyText}>No violations</Text>
        )}
      </View>

      {/* Recent Events */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>Recent Events ({events.length})</Text>
        {events.slice(0, 10).map((event) => (
          <View key={event.id} style={styles.eventCard}>
            <View style={styles.eventHeader}>
              <Text style={styles.eventPlate}>{event.plate_text}</Text>
              <View
                style={[
                  styles.eventBadge,
                  event.result === 'approved' ? styles.approvedBadge : styles.violationBadge,
                ]}
              >
                <Text style={styles.eventBadgeText}>{event.result}</Text>
              </View>
            </View>
            <Text style={styles.eventDetails}>
              {event.location} • Conf: {(event.confidence * 100).toFixed(0)}%
            </Text>
            {event.notes && <Text style={styles.eventNotes}>{event.notes}</Text>}
            <Text style={styles.eventTime}>{formatTimestamp(event.timestamp)}</Text>
          </View>
        ))}
        {events.length === 0 && (
          <Text style={styles.emptyText}>No events yet</Text>
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
  card: {
    backgroundColor: '#1a1a1a',
    margin: 12,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#333',
  },
  cardHeader: {
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
  input: {
    backgroundColor: '#000',
    borderWidth: 1,
    borderColor: '#333',
    borderRadius: 8,
    padding: 12,
    color: '#FFF',
    fontSize: 16,
    marginBottom: 12,
  },
  inputRow: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 12,
  },
  label: {
    color: '#888',
    fontSize: 14,
    marginBottom: 8,
  },
  radioGroup: {
    marginBottom: 12,
  },
  radioButtons: {
    flexDirection: 'row',
    gap: 12,
  },
  radioButton: {
    flex: 1,
    paddingVertical: 10,
    paddingHorizontal: 16,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#333',
    alignItems: 'center',
  },
  radioButtonActive: {
    backgroundColor: '#007AFF',
    borderColor: '#007AFF',
  },
  radioText: {
    color: '#888',
    fontSize: 14,
    fontWeight: '600',
  },
  radioTextActive: {
    color: '#FFF',
  },
  primaryButton: {
    backgroundColor: '#007AFF',
    padding: 14,
    borderRadius: 8,
    alignItems: 'center',
  },
  secondaryButton: {
    backgroundColor: '#4CAF50',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
    marginBottom: 12,
  },
  addButton: {
    backgroundColor: '#4CAF50',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    minWidth: 60,
  },
  resetButton: {
    backgroundColor: '#FF3B30',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 6,
  },
  resetButtonText: {
    color: '#FFF',
    fontSize: 12,
    fontWeight: '600',
  },
  buttonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: '600',
  },
  listContainer: {
    maxHeight: 120,
  },
  permitCard: {
    backgroundColor: '#000',
    padding: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#333',
    marginRight: 8,
    minWidth: 120,
  },
  permitPlate: {
    color: '#FFD700',
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  permitType: {
    color: '#888',
    fontSize: 12,
  },
  stayCard: {
    backgroundColor: '#000',
    padding: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#333',
    marginBottom: 8,
  },
  stayPlate: {
    color: '#FFD700',
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  stayTime: {
    color: '#FFA500',
    fontSize: 14,
    marginBottom: 2,
  },
  stayTimestamp: {
    color: '#666',
    fontSize: 11,
  },
  violationCard: {
    backgroundColor: '#000',
    padding: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#FF3B30',
    marginBottom: 8,
  },
  violationPlate: {
    color: '#FFD700',
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  violationReason: {
    color: '#FF6B6B',
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 2,
    textTransform: 'capitalize',
  },
  violationLocation: {
    color: '#888',
    fontSize: 12,
    marginBottom: 2,
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
    marginBottom: 8,
  },
  eventHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  eventPlate: {
    color: '#FFD700',
    fontSize: 16,
    fontWeight: 'bold',
  },
  eventBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  approvedBadge: {
    backgroundColor: '#4CAF50',
  },
  violationBadge: {
    backgroundColor: '#FF3B30',
  },
  eventBadgeText: {
    color: '#FFF',
    fontSize: 11,
    fontWeight: '700',
    textTransform: 'uppercase',
  },
  eventDetails: {
    color: '#888',
    fontSize: 12,
    marginBottom: 2,
  },
  eventNotes: {
    color: '#AAA',
    fontSize: 11,
    fontStyle: 'italic',
    marginBottom: 2,
  },
  eventTime: {
    color: '#666',
    fontSize: 11,
  },
  emptyText: {
    color: '#666',
    fontSize: 14,
    textAlign: 'center',
    paddingVertical: 20,
  },
});
