import { StyleSheet, ScrollView, TouchableOpacity, View, TextInput, Alert, Modal } from 'react-native';
import { useState, useCallback } from 'react';
import { useFocusEffect } from 'expo-router';

import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';

const BACKEND_URL = 'http://10.0.0.67:8000'; // Update with your computer's IP

const US_STATES = [
  'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
  'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
  'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
  'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
  'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
];

interface Permit {
  id: number;
  plate_text: string;
  permit_type: string;
  state?: string;
  notes?: string;
}

export default function HomeScreen() {
  // Manual entry state
  const [plateInput, setPlateInput] = useState('');
  const [selectedState, setSelectedState] = useState('IN');
  const [showStatePicker, setShowStatePicker] = useState(false);
  
  // Permits state
  const [permits, setPermits] = useState<Permit[]>([]);
  const [newPermitPlate, setNewPermitPlate] = useState('');
  const [newPermitState, setNewPermitState] = useState('IN');
  const [showPermitStatePicker, setShowPermitStatePicker] = useState(false);

  const loadPermits = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/permits`);
      if (response.ok) {
        setPermits(await response.json());
      }
    } catch (error) {
      console.error('Error loading permits:', error);
    }
  };

  useFocusEffect(
    useCallback(() => {
      loadPermits();
    }, [])
  );

  const submitManualEntry = async () => {
    if (!plateInput.trim()) {
      Alert.alert('Error', 'Please enter a plate number');
      return;
    }

    try {
      // Submit to enforcement API - will auto-check against permits
      const response = await fetch(`${BACKEND_URL}/api/ocr_event`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          plate_text: plateInput.toUpperCase(),
          confidence: 1.0, // Manual entry is 100% confident
          timestamp: new Date().toISOString(),
          location: 'timed', // Will auto-detect if it's a permit
          state: selectedState,
        }),
      });

      const result = await response.json();
      
      if (result.result === 'approved') {
        Alert.alert('✅ Entry Logged', result.message);
      } else {
        Alert.alert('⚠️ Entry Logged', result.message);
      }

      setPlateInput('');
    } catch (error) {
      Alert.alert('Error', 'Failed to submit entry: ' + String(error));
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
          state: newPermitState,
        }),
      });

      if (response.ok) {
        Alert.alert('Success', 'Permit added');
        setNewPermitPlate('');
        await loadPermits();
      } else {
        const error = await response.json();
        Alert.alert('Error', error.detail || 'Failed to add permit');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to add permit: ' + String(error));
    }
  };

  const deletePermit = async (permitId: number) => {
    Alert.alert(
      'Delete Permit',
      'Are you sure you want to delete this permit?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              await fetch(`${BACKEND_URL}/api/permits/${permitId}`, {
                method: 'DELETE',
              });
              await loadPermits();
            } catch (error) {
              Alert.alert('Error', 'Failed to delete permit');
            }
          },
        },
      ]
    );
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <ThemedText type="title">Parking Management</ThemedText>
        <ThemedText style={styles.subtitle}>Manual entry and permit management</ThemedText>
      </View>

      {/* Manual OCR Entry Section */}
      <View style={styles.card}>
        <ThemedText style={styles.cardTitle}>Manual License Plate Entry</ThemedText>
        
        <View style={styles.inputGroup}>
          <ThemedText style={styles.label}>License Plate Number</ThemedText>
          <TextInput
            style={styles.input}
            placeholder="e.g., ABC123"
            placeholderTextColor="#666"
            value={plateInput}
            onChangeText={setPlateInput}
            autoCapitalize="characters"
          />
        </View>

        <View style={styles.inputGroup}>
          <ThemedText style={styles.label}>State</ThemedText>
          <TouchableOpacity 
            style={styles.stateSelector}
            onPress={() => setShowStatePicker(true)}
          >
            <ThemedText style={styles.stateSelectorText}>{selectedState}</ThemedText>
            <ThemedText style={styles.stateSelectorArrow}>▼</ThemedText>
          </TouchableOpacity>
        </View>

        <TouchableOpacity style={styles.primaryButton} onPress={submitManualEntry}>
          <ThemedText style={styles.buttonText}>Submit Entry</ThemedText>
        </TouchableOpacity>
      </View>

      {/* Manage Permits Section */}
      <View style={styles.card}>
        <ThemedText style={styles.cardTitle}>Manage Parking Permits ({permits.length})</ThemedText>
        
        <View style={styles.addPermitRow}>
          <View style={styles.permitInputContainer}>
            <TextInput
              style={[styles.input, styles.permitPlateInput]}
              placeholder="Plate Number"
              placeholderTextColor="#666"
              value={newPermitPlate}
              onChangeText={setNewPermitPlate}
              autoCapitalize="characters"
            />
            <TouchableOpacity 
              style={styles.permitStateSelector}
              onPress={() => setShowPermitStatePicker(true)}
            >
              <ThemedText style={styles.permitStateSelectorText}>{newPermitState}</ThemedText>
            </TouchableOpacity>
          </View>
          <TouchableOpacity style={styles.addButton} onPress={addPermit}>
            <ThemedText style={styles.buttonText}>Add</ThemedText>
          </TouchableOpacity>
        </View>

        <View style={styles.permitsList}>
          {permits.map((permit) => (
            <View key={permit.id} style={styles.permitCard}>
              <View style={styles.permitInfo}>
                <ThemedText style={styles.permitPlate}>{permit.plate_text}</ThemedText>
                <ThemedText style={styles.permitState}>{permit.state || 'N/A'}</ThemedText>
              </View>
              <TouchableOpacity
                style={styles.deleteButton}
                onPress={() => deletePermit(permit.id)}
              >
                <ThemedText style={styles.deleteButtonText}>×</ThemedText>
              </TouchableOpacity>
            </View>
          ))}
          {permits.length === 0 && (
            <ThemedText style={styles.emptyText}>No permits added yet</ThemedText>
          )}
        </View>
      </View>

      {/* State Picker Modal for Manual Entry */}
      <Modal
        visible={showStatePicker}
        transparent={true}
        animationType="slide"
        onRequestClose={() => setShowStatePicker(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <ThemedText style={styles.modalTitle}>Select State</ThemedText>
            <ScrollView style={styles.stateList}>
              {US_STATES.map(state => (
                <TouchableOpacity
                  key={state}
                  style={styles.stateOption}
                  onPress={() => {
                    setSelectedState(state);
                    setShowStatePicker(false);
                  }}
                >
                  <ThemedText style={[
                    styles.stateOptionText,
                    selectedState === state && styles.stateOptionTextSelected
                  ]}>
                    {state}
                  </ThemedText>
                </TouchableOpacity>
              ))}
            </ScrollView>
            <TouchableOpacity 
              style={styles.modalCloseButton}
              onPress={() => setShowStatePicker(false)}
            >
              <ThemedText style={styles.buttonText}>Close</ThemedText>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      {/* State Picker Modal for Permit */}
      <Modal
        visible={showPermitStatePicker}
        transparent={true}
        animationType="slide"
        onRequestClose={() => setShowPermitStatePicker(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <ThemedText style={styles.modalTitle}>Select State</ThemedText>
            <ScrollView style={styles.stateList}>
              {US_STATES.map(state => (
                <TouchableOpacity
                  key={state}
                  style={styles.stateOption}
                  onPress={() => {
                    setNewPermitState(state);
                    setShowPermitStatePicker(false);
                  }}
                >
                  <ThemedText style={[
                    styles.stateOptionText,
                    newPermitState === state && styles.stateOptionTextSelected
                  ]}>
                    {state}
                  </ThemedText>
                </TouchableOpacity>
              ))}
            </ScrollView>
            <TouchableOpacity 
              style={styles.modalCloseButton}
              onPress={() => setShowPermitStatePicker(false)}
            >
              <ThemedText style={styles.buttonText}>Close</ThemedText>
            </TouchableOpacity>
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
  subtitle: {
    fontSize: 14,
    color: '#888',
    marginTop: 4,
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
    marginBottom: 16,
  },
  inputGroup: {
    marginBottom: 16,
  },
  label: {
    color: '#888',
    fontSize: 14,
    marginBottom: 8,
    fontWeight: '600',
  },
  input: {
    backgroundColor: '#000',
    borderWidth: 1,
    borderColor: '#333',
    borderRadius: 8,
    padding: 12,
    color: '#FFF',
    fontSize: 16,
  },
  stateSelector: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#000',
    borderWidth: 1,
    borderColor: '#333',
    borderRadius: 8,
    padding: 12,
  },
  stateSelectorText: {
    color: '#FFF',
    fontSize: 16,
  },
  stateSelectorArrow: {
    color: '#888',
    fontSize: 12,
  },
  primaryButton: {
    backgroundColor: '#007AFF',
    padding: 14,
    borderRadius: 8,
    alignItems: 'center',
  },
  buttonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: '600',
  },
  addPermitRow: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 16,
  },
  permitInputContainer: {
    flex: 1,
    flexDirection: 'row',
    gap: 8,
  },
  permitPlateInput: {
    flex: 2,
    marginBottom: 0,
  },
  permitStateSelector: {
    flex: 1,
    backgroundColor: '#000',
    borderWidth: 1,
    borderColor: '#333',
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 8,
  },
  permitStateSelectorText: {
    color: '#FFF',
    fontSize: 14,
    fontWeight: '600',
  },
  addButton: {
    backgroundColor: '#4CAF50',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    minWidth: 60,
  },
  permitsList: {
    gap: 8,
  },
  permitCard: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#000',
    padding: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#333',
  },
  permitInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    flex: 1,
  },
  permitPlate: {
    color: '#FFD700',
    fontSize: 16,
    fontWeight: 'bold',
    fontFamily: 'monospace',
    letterSpacing: 1,
  },
  permitState: {
    color: '#FFF',
    fontSize: 14,
    backgroundColor: '#007AFF',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
    fontWeight: '700',
  },
  deleteButton: {
    padding: 4,
  },
  deleteButtonText: {
    color: '#FF3B30',
    fontSize: 24,
    fontWeight: 'bold',
  },
  emptyText: {
    color: '#666',
    fontSize: 14,
    textAlign: 'center',
    paddingVertical: 20,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    backgroundColor: '#1a1a1a',
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    padding: 20,
    maxHeight: '70%',
  },
  modalTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFF',
    marginBottom: 16,
    textAlign: 'center',
  },
  stateList: {
    maxHeight: 400,
  },
  stateOption: {
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  stateOptionText: {
    color: '#FFF',
    fontSize: 16,
  },
  stateOptionTextSelected: {
    color: '#007AFF',
    fontWeight: 'bold',
  },
  modalCloseButton: {
    backgroundColor: '#333',
    padding: 14,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 16,
  },
});
