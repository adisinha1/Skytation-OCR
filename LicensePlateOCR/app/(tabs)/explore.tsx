import React, { useState, useCallback, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  TextInput,
  Alert,
  Modal,
} from 'react-native';
import MapView, { Marker, Circle, Region } from 'react-native-maps';
import * as Location from 'expo-location';
import { useFocusEffect } from 'expo-router';
import {
  getZones,
  saveZone,
  deleteZone,
  clearAllZones,
  CampusZone,
} from '@/app/campusZones';

export default function ZonesScreen() {
  const mapRef = useRef<MapView>(null);
  const [zones, setZones] = useState<CampusZone[]>([]);
  const [showAddModal, setShowAddModal] = useState(false);
  const [selectedLocation, setSelectedLocation] = useState<{
    latitude: number;
    longitude: number;
  } | null>(null);
  
  // Form state
  const [zoneName, setZoneName] = useState('');
  const [zoneCode, setZoneCode] = useState('');

  // Map region (centered on your campus - update these coordinates!)
  const [mapRegion, setMapRegion] = useState<Region>({
    latitude: 40.4259, // Purdue University approximate center
    longitude: -86.9081,
    latitudeDelta: 0.01, // Zoom level
    longitudeDelta: 0.01,
  });

  const loadZones = async () => {
    const loadedZones = await getZones();
    setZones(loadedZones);
  };

  const centerOnCurrentLocation = async () => {
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission Denied', 'Location permission is required');
        return;
      }

      const location = await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.High,
      });

      const newRegion = {
        latitude: location.coords.latitude,
        longitude: location.coords.longitude,
        latitudeDelta: 0.01,
        longitudeDelta: 0.01,
      };

      setMapRegion(newRegion);
      mapRef.current?.animateToRegion(newRegion, 1000);
    } catch (error) {
      Alert.alert('Error', 'Could not get current location');
    }
  };

  const handleMapPress = (event: any) => {
    const { latitude, longitude } = event.nativeEvent.coordinate;
    setSelectedLocation({ latitude, longitude });
    setShowAddModal(true);
  };

  const handleSaveZone = async () => {
    if (!zoneName.trim()) {
      Alert.alert('Error', 'Please enter a zone name');
      return;
    }
    if (!zoneCode.trim()) {
      Alert.alert('Error', 'Please enter a zone code');
      return;
    }
    if (!selectedLocation) {
      Alert.alert('Error', 'Please select a location on the map');
      return;
    }

    await saveZone({
      name: zoneName,
      code: zoneCode.toUpperCase(),
      latitude: selectedLocation.latitude,
      longitude: selectedLocation.longitude,
      radius: 0.0005,
    });

    // Reset form
    setZoneName('');
    setZoneCode('');
    setSelectedLocation(null);
    setShowAddModal(false);
    
    // Reload zones
    await loadZones();
    
    Alert.alert('Success', 'Zone saved successfully!');
  };

  const handleDeleteZone = (zone: CampusZone) => {
    Alert.alert(
      'Delete Zone',
      `Are you sure you want to delete ${zone.name}?`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            await deleteZone(zone.id);
            await loadZones();
          },
        },
      ]
    );
  };

  const handleClearAll = () => {
    Alert.alert(
      'Clear All Zones',
      'Are you sure you want to delete all zones?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear All',
          style: 'destructive',
          onPress: async () => {
            await clearAllZones();
            await loadZones();
          },
        },
      ]
    );
  };

  const focusOnZone = (zone: CampusZone) => {
    const region = {
      latitude: zone.latitude,
      longitude: zone.longitude,
      latitudeDelta: 0.005,
      longitudeDelta: 0.005,
    };
    mapRef.current?.animateToRegion(region, 1000);
  };

  useFocusEffect(
    useCallback(() => {
      loadZones();
    }, [])
  );

  return (
    <View style={styles.container}>
      {/* Map View */}
      <View style={styles.mapContainer}>
        <MapView
          ref={mapRef}
          style={styles.map}
          initialRegion={mapRegion}
          onPress={handleMapPress}
          showsUserLocation
          showsMyLocationButton
        >
          {/* Existing zones */}
          {zones.map((zone) => (
            <React.Fragment key={zone.id}>
              <Marker
                coordinate={{
                  latitude: zone.latitude,
                  longitude: zone.longitude,
                }}
                title={zone.code}
                description={zone.name}
                pinColor="#FF6B35"
              />
              <Circle
                center={{
                  latitude: zone.latitude,
                  longitude: zone.longitude,
                }}
                radius={50} // 50 meters
                strokeColor="rgba(255, 107, 53, 0.5)"
                fillColor="rgba(255, 107, 53, 0.1)"
              />
            </React.Fragment>
          ))}

          {/* Selected location (preview) */}
          {selectedLocation && (
            <>
              <Marker
                coordinate={selectedLocation}
                pinColor="#00FF00"
                title="New Zone"
              />
              <Circle
                center={selectedLocation}
                radius={50}
                strokeColor="rgba(0, 255, 0, 0.5)"
                fillColor="rgba(0, 255, 0, 0.1)"
              />
            </>
          )}
        </MapView>

        {/* Map Controls */}
        <View style={styles.mapControls}>
          <TouchableOpacity
            style={styles.controlButton}
            onPress={centerOnCurrentLocation}
          >
            <Text style={styles.controlButtonText}>📍 My Location</Text>
          </TouchableOpacity>
        </View>

        {/* Instructions */}
        <View style={styles.instructions}>
          <Text style={styles.instructionsText}>
            Tap anywhere on the map to add a new parking zone
          </Text>
        </View>
      </View>

      {/* Zones List */}
      <View style={styles.zonesListContainer}>
        <View style={styles.zonesHeader}>
          <Text style={styles.zonesTitle}>Saved Zones ({zones.length})</Text>
          {zones.length > 0 && (
            <TouchableOpacity
              style={styles.clearAllButton}
              onPress={handleClearAll}
            >
              <Text style={styles.clearAllButtonText}>Clear All</Text>
            </TouchableOpacity>
          )}
        </View>

        <ScrollView style={styles.zonesList} horizontal>
          {zones.length === 0 ? (
            <View style={styles.emptyState}>
              <Text style={styles.emptyText}>No zones yet</Text>
            </View>
          ) : (
            zones.map((zone) => (
              <TouchableOpacity
                key={zone.id}
                style={styles.zoneCard}
                onPress={() => focusOnZone(zone)}
              >
                <View style={styles.zoneCardHeader}>
                  <View style={styles.zoneCodeBadge}>
                    <Text style={styles.zoneCodeText}>{zone.code}</Text>
                  </View>
                  <TouchableOpacity
                    style={styles.deleteIconButton}
                    onPress={() => handleDeleteZone(zone)}
                  >
                    <Text style={styles.deleteIcon}>×</Text>
                  </TouchableOpacity>
                </View>
                <Text style={styles.zoneCardName} numberOfLines={2}>
                  {zone.name}
                </Text>
                <Text style={styles.zoneCardCoords}>
                  {zone.latitude.toFixed(4)}, {zone.longitude.toFixed(4)}
                </Text>
              </TouchableOpacity>
            ))
          )}
        </ScrollView>
      </View>

      {/* Add Zone Modal */}
      <Modal
        visible={showAddModal}
        animationType="slide"
        transparent={true}
        onRequestClose={() => setShowAddModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Add New Zone</Text>

            {selectedLocation && (
              <View style={styles.locationPreview}>
                <Text style={styles.locationLabel}>Selected Location:</Text>
                <Text style={styles.coordinates}>
                  {selectedLocation.latitude.toFixed(6)},{' '}
                  {selectedLocation.longitude.toFixed(6)}
                </Text>
              </View>
            )}

            <View style={styles.inputGroup}>
              <Text style={styles.label}>Zone Name</Text>
              <TextInput
                style={styles.input}
                placeholder="e.g., Parking Lot A - North"
                placeholderTextColor="#666"
                value={zoneName}
                onChangeText={setZoneName}
              />
            </View>

            <View style={styles.inputGroup}>
              <Text style={styles.label}>Zone Code</Text>
              <TextInput
                style={styles.input}
                placeholder="e.g., A1"
                placeholderTextColor="#666"
                value={zoneCode}
                onChangeText={setZoneCode}
                maxLength={4}
                autoCapitalize="characters"
              />
            </View>

            <View style={styles.modalButtons}>
              <TouchableOpacity
                style={[styles.button, styles.cancelButton]}
                onPress={() => {
                  setShowAddModal(false);
                  setSelectedLocation(null);
                  setZoneName('');
                  setZoneCode('');
                }}
              >
                <Text style={styles.buttonText}>Cancel</Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[styles.button, styles.saveButton]}
                onPress={handleSaveZone}
              >
                <Text style={styles.buttonText}>💾 Save Zone</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  mapContainer: {
    flex: 1,
    position: 'relative',
  },
  map: {
    flex: 1,
  },
  mapControls: {
    position: 'absolute',
    top: 60,
    right: 16,
    gap: 8,
  },
  controlButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 8,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5,
  },
  controlButtonText: {
    color: '#FFF',
    fontSize: 14,
    fontWeight: '600',
  },
  instructions: {
    position: 'absolute',
    bottom: 16,
    left: 16,
    right: 16,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    padding: 12,
    borderRadius: 8,
  },
  instructionsText: {
    color: '#FFF',
    fontSize: 13,
    textAlign: 'center',
  },
  zonesListContainer: {
    height: 140,
    backgroundColor: '#1a1a1a',
    borderTopWidth: 1,
    borderTopColor: '#333',
  },
  zonesHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 12,
    paddingBottom: 8,
  },
  zonesTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#FFF',
  },
  clearAllButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: '#FF3B30',
    borderRadius: 6,
  },
  clearAllButtonText: {
    color: '#FFF',
    fontSize: 11,
    fontWeight: '600',
  },
  zonesList: {
    paddingHorizontal: 12,
  },
  emptyState: {
    padding: 20,
    alignItems: 'center',
  },
  emptyText: {
    fontSize: 14,
    color: '#666',
  },
  zoneCard: {
    backgroundColor: '#000',
    borderRadius: 8,
    padding: 12,
    marginRight: 12,
    width: 160,
    borderWidth: 1,
    borderColor: '#333',
  },
  zoneCardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  zoneCodeBadge: {
    backgroundColor: '#FF6B35',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  zoneCodeText: {
    color: '#FFF',
    fontSize: 12,
    fontWeight: 'bold',
  },
  deleteIconButton: {
    padding: 4,
  },
  deleteIcon: {
    color: '#FF3B30',
    fontSize: 24,
    fontWeight: 'bold',
  },
  zoneCardName: {
    fontSize: 12,
    fontWeight: '600',
    color: '#FFF',
    marginBottom: 4,
  },
  zoneCardCoords: {
    fontSize: 10,
    color: '#666',
    fontFamily: 'monospace',
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
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
  locationPreview: {
    backgroundColor: '#000',
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
    borderLeftWidth: 3,
    borderLeftColor: '#00FF00',
  },
  locationLabel: {
    color: '#888',
    fontSize: 11,
    marginBottom: 4,
  },
  coordinates: {
    color: '#00FF00',
    fontSize: 13,
    fontFamily: 'monospace',
  },
  inputGroup: {
    marginBottom: 16,
  },
  label: {
    fontSize: 12,
    color: '#888',
    marginBottom: 8,
    textTransform: 'uppercase',
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
  modalButtons: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 8,
  },
  button: {
    flex: 1,
    padding: 14,
    borderRadius: 8,
    alignItems: 'center',
  },
  cancelButton: {
    backgroundColor: '#333',
  },
  saveButton: {
    backgroundColor: '#4CAF50',
  },
  buttonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: '600',
  },
});