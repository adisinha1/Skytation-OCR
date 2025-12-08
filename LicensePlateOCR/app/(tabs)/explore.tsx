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
  Platform,
} from 'react-native';
import * as Location from 'expo-location';
import { useFocusEffect } from 'expo-router';

const BACKEND_URL = 'http://10.0.0.66:8000'; // Update with your computer's IP

// Only import WebView on native platforms
let WebView: any = null;
if (Platform.OS !== 'web') {
  WebView = require('react-native-webview').WebView;
}

interface Zone {
  id: number;
  name: string;
  code: string;
  latitude: number;
  longitude: number;
  radius: number;
  zone_type: string;
  default_time_limit: number;
}

export default function ZonesScreen() {
  const webViewRef = useRef<any>(null);
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [zones, setZones] = useState<Zone[]>([]);
  const [showAddModal, setShowAddModal] = useState(false);
  const [selectedLocation, setSelectedLocation] = useState<{
    latitude: number;
    longitude: number;
  } | null>(null);
  const [mapKey, setMapKey] = useState(0);
  
  // Form state
  const [zoneName, setZoneName] = useState('');
  const [zoneCode, setZoneCode] = useState('');
  const [zoneType, setZoneType] = useState<'permit' | 'timed'>('timed');
  const [defaultTimeLimit, setDefaultTimeLimit] = useState(120);
  
  // Map center (default: Purdue University area)
  const [mapCenter, setMapCenter] = useState({
    latitude: 40.4259,
    longitude: -86.9081,
  });

  const loadZones = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/zones`);
      if (response.ok) {
        const loadedZones = await response.json();
        setZones(loadedZones);
        setMapKey(prev => prev + 1);
      }
    } catch (error) {
      console.error('Error loading zones:', error);
    }
  };

  useFocusEffect(
    useCallback(() => {
      loadZones();
    }, [])
  );

  // Generate the Leaflet map HTML
  const generateMapHTML = () => {
    const markersJS = zones.map(zone => `
      L.marker([${zone.latitude}, ${zone.longitude}])
        .addTo(map)
        .bindPopup('<b>${zone.name}</b><br>${zone.code}<br>${zone.zone_type} - ${zone.default_time_limit}min');
      L.circle([${zone.latitude}, ${zone.longitude}], {
        color: '${zone.zone_type === 'permit' ? '#007AFF' : '#4CAF50'}',
        fillColor: '${zone.zone_type === 'permit' ? '#007AFF' : '#4CAF50'}',
        fillOpacity: 0.2,
        radius: 50
      }).addTo(map);
    `).join('\n');

    return `
      <!DOCTYPE html>
      <html>
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
          * { margin: 0; padding: 0; }
          html, body, #map { height: 100%; width: 100%; }
        </style>
      </head>
      <body>
        <div id="map"></div>
        <script>
          var map = L.map('map').setView([${mapCenter.latitude}, ${mapCenter.longitude}], 16);
          
          L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors'
          }).addTo(map);
          
          ${markersJS}
          
          map.on('click', function(e) {
            var message = JSON.stringify({
              type: 'mapClick',
              latitude: e.latlng.lat,
              longitude: e.latlng.lng
            });
            
            if (window.ReactNativeWebView) {
              window.ReactNativeWebView.postMessage(message);
            } else if (window.parent !== window) {
              window.parent.postMessage(message, '*');
            }
          });
          
          function centerMap(lat, lng) {
            map.setView([lat, lng], 16);
          }
          
          var tempMarker = null;
          function showTempMarker(lat, lng) {
            if (tempMarker) {
              map.removeLayer(tempMarker);
            }
            tempMarker = L.marker([lat, lng], {
              icon: L.divIcon({
                className: 'temp-marker',
                html: '<div style="background: #FF3B30; width: 20px; height: 20px; border-radius: 50%; border: 3px solid white;"></div>',
                iconSize: [20, 20],
                iconAnchor: [10, 10]
              })
            }).addTo(map);
          }
          
          window.addEventListener('message', function(event) {
            try {
              var data = JSON.parse(event.data);
              if (data.type === 'centerMap') {
                centerMap(data.latitude, data.longitude);
              } else if (data.type === 'showTempMarker') {
                showTempMarker(data.latitude, data.longitude);
              }
            } catch (e) {}
          });
        </script>
      </body>
      </html>
    `;
  };

  const handleMapMessage = (data: any) => {
    if (data.type === 'mapClick') {
      setSelectedLocation({
        latitude: data.latitude,
        longitude: data.longitude,
      });
      setShowAddModal(true);
      
      if (Platform.OS === 'web') {
        iframeRef.current?.contentWindow?.postMessage(
          JSON.stringify({ type: 'showTempMarker', latitude: data.latitude, longitude: data.longitude }),
          '*'
        );
      } else {
        webViewRef.current?.injectJavaScript(`
          showTempMarker(${data.latitude}, ${data.longitude});
          true;
        `);
      }
    }
  };

  const handleWebViewMessage = (event: any) => {
    try {
      const data = JSON.parse(event.nativeEvent.data);
      handleMapMessage(data);
    } catch (e) {
      console.error('Error parsing WebView message:', e);
    }
  };

  React.useEffect(() => {
    if (Platform.OS === 'web') {
      const handleMessage = (event: MessageEvent) => {
        try {
          const data = JSON.parse(event.data);
          handleMapMessage(data);
        } catch (e) {}
      };
      
      window.addEventListener('message', handleMessage);
      return () => window.removeEventListener('message', handleMessage);
    }
  }, []);

  const centerOnCurrentLocation = async () => {
    try {
      const { status } = await Location.requestForegroundPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('Permission denied', 'Location permission is required');
        return;
      }

      const location = await Location.getCurrentPositionAsync({});
      const { latitude, longitude } = location.coords;
      
      setMapCenter({ latitude, longitude });
      
      if (Platform.OS === 'web') {
        iframeRef.current?.contentWindow?.postMessage(
          JSON.stringify({ type: 'centerMap', latitude, longitude }),
          '*'
        );
      } else {
        webViewRef.current?.injectJavaScript(`
          centerMap(${latitude}, ${longitude});
          true;
        `);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to get current location');
    }
  };

  const handleAddZone = async () => {
    if (!zoneName.trim() || !zoneCode.trim()) {
      Alert.alert('Error', 'Please fill in all fields');
      return;
    }

    if (!selectedLocation) {
      Alert.alert('Error', 'Please select a location on the map');
      return;
    }

    try {
      const response = await fetch(`${BACKEND_URL}/api/zones`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: zoneName,
          code: zoneCode,
          latitude: selectedLocation.latitude,
          longitude: selectedLocation.longitude,
          radius: 0.0005,
          zone_type: zoneType,
          default_time_limit: defaultTimeLimit,
        }),
      });

      if (response.ok) {
        setZoneName('');
        setZoneCode('');
        setZoneType('timed');
        setDefaultTimeLimit(120);
        setSelectedLocation(null);
        setShowAddModal(false);
        await loadZones();
      } else {
        const error = await response.json();
        Alert.alert('Error', error.detail || 'Failed to add zone');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to add zone: ' + String(error));
    }
  };

  const handleDeleteZone = async (id: number) => {
    Alert.alert(
      'Delete Zone',
      'Are you sure you want to delete this zone?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            try {
              const response = await fetch(`${BACKEND_URL}/api/zones/${id}`, {
                method: 'DELETE',
              });
              if (response.ok) {
                await loadZones();
              } else {
                Alert.alert('Error', 'Failed to delete zone');
              }
            } catch (error) {
              Alert.alert('Error', 'Failed to delete zone: ' + String(error));
            }
          },
        },
      ]
    );
  };

  const handleClearAll = async () => {
    Alert.alert(
      'Clear All Zones',
      'Are you sure you want to delete all zones?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Clear All',
          style: 'destructive',
          onPress: async () => {
            try {
              const response = await fetch(`${BACKEND_URL}/api/zones/clear`, {
                method: 'DELETE',
              });
              if (response.ok) {
                await loadZones();
              }
            } catch (error) {
              Alert.alert('Error', 'Failed to clear zones');
            }
          },
        },
      ]
    );
  };

  const handleSeedZones = async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/zones/seed`, {
        method: 'POST',
      });
      if (response.ok) {
        const result = await response.json();
        Alert.alert('Success', result.message);
        await loadZones();
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to seed zones');
    }
  };

  const renderMap = () => {
    if (Platform.OS === 'web') {
      return (
        <iframe
          ref={iframeRef}
          key={mapKey}
          srcDoc={generateMapHTML()}
          style={{
            width: '100%',
            height: '100%',
            border: 'none',
          }}
        />
      );
    } else {
      return (
        <WebView
          ref={webViewRef}
          key={mapKey}
          source={{ html: generateMapHTML() }}
          style={styles.map}
          onMessage={handleWebViewMessage}
          javaScriptEnabled={true}
          domStorageEnabled={true}
          startInLoadingState={true}
          scalesPageToFit={true}
        />
      );
    }
  };

  return (
    <View style={styles.container}>
      {/* Map */}
      <View style={styles.mapContainer}>
        {renderMap()}
        
        {/* Map Controls Overlay */}
        <View style={styles.mapControls}>
          <TouchableOpacity
            style={styles.locationButton}
            onPress={centerOnCurrentLocation}
          >
            <Text style={styles.buttonIcon}>📍</Text>
          </TouchableOpacity>
        </View>
        
        {/* Instructions Overlay */}
        <View style={styles.instructionsOverlay}>
          <Text style={styles.instructionsText}>Tap on map to add a zone</Text>
        </View>
      </View>

      {/* Zone List */}
      <View style={styles.zoneListContainer}>
        <View style={styles.zoneListHeader}>
          <Text style={styles.zoneListTitle}>Zones ({zones.length})</Text>
          <View style={styles.headerButtons}>
            {zones.length === 0 && (
              <TouchableOpacity
                style={styles.seedButton}
                onPress={handleSeedZones}
              >
                <Text style={styles.seedButtonText}>Seed Sample</Text>
              </TouchableOpacity>
            )}
            {zones.length > 0 && (
              <TouchableOpacity
                style={styles.clearAllButton}
                onPress={handleClearAll}
              >
                <Text style={styles.clearAllButtonText}>Clear All</Text>
              </TouchableOpacity>
            )}
          </View>
        </View>

        <ScrollView 
          style={styles.zoneList}
          showsVerticalScrollIndicator={true}
        >
          {zones.length === 0 ? (
            <Text style={styles.emptyText}>
              No zones configured. Tap on the map to add one or seed sample zones.
            </Text>
          ) : (
            zones.map((zone) => (
              <View key={zone.id} style={styles.zoneCard}>
                <View style={styles.zoneInfo}>
                  <Text style={styles.zoneName}>{zone.name}</Text>
                  <View style={styles.zoneMetaRow}>
                    <Text style={styles.zoneCode}>{zone.code}</Text>
                    <Text style={[
                      styles.zoneType,
                      zone.zone_type === 'permit' ? styles.zoneTypePermit : styles.zoneTypeTimed
                    ]}>
                      {zone.zone_type}
                    </Text>
                    {zone.zone_type === 'timed' && (
                      <Text style={styles.zoneTimeLimit}>{zone.default_time_limit}min</Text>
                    )}
                  </View>
                  <Text style={styles.zoneCoords}>
                    {zone.latitude.toFixed(5)}, {zone.longitude.toFixed(5)}
                  </Text>
                </View>
                <TouchableOpacity
                  style={styles.deleteButton}
                  onPress={() => handleDeleteZone(zone.id)}
                >
                  <Text style={styles.deleteButtonText}>×</Text>
                </TouchableOpacity>
              </View>
            ))
          )}
        </ScrollView>
      </View>

      {/* Add Zone Modal */}
      <Modal
        visible={showAddModal}
        transparent={true}
        animationType="slide"
        onRequestClose={() => {
          setShowAddModal(false);
          setSelectedLocation(null);
        }}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Add New Zone</Text>

            <View style={styles.inputGroup}>
              <Text style={styles.inputLabel}>Zone Name</Text>
              <TextInput
                style={styles.input}
                placeholder="e.g., Parking Lot A"
                placeholderTextColor="#666"
                value={zoneName}
                onChangeText={setZoneName}
              />
            </View>

            <View style={styles.inputGroup}>
              <Text style={styles.inputLabel}>Zone Code</Text>
              <TextInput
                style={styles.input}
                placeholder="e.g., A1"
                placeholderTextColor="#666"
                value={zoneCode}
                onChangeText={setZoneCode}
              />
            </View>

            {/* Zone Type Toggle */}
            <View style={styles.inputGroup}>
              <Text style={styles.inputLabel}>Zone Type</Text>
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

            {/* Time Limit (for timed zones) */}
            {zoneType === 'timed' && (
              <View style={styles.inputGroup}>
                <Text style={styles.inputLabel}>Default Time Limit (minutes)</Text>
                <View style={styles.timeLimitOptions}>
                  {[30, 60, 120, 240].map((mins) => (
                    <TouchableOpacity
                      key={mins}
                      style={[
                        styles.timeLimitButton,
                        defaultTimeLimit === mins && styles.timeLimitButtonActive
                      ]}
                      onPress={() => setDefaultTimeLimit(mins)}
                    >
                      <Text style={[
                        styles.timeLimitButtonText,
                        defaultTimeLimit === mins && styles.timeLimitButtonTextActive
                      ]}>
                        {mins < 60 ? `${mins}m` : `${mins/60}h`}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>
              </View>
            )}

            {selectedLocation && (
              <View style={styles.coordsDisplay}>
                <Text style={styles.coordsLabel}>Location</Text>
                <Text style={styles.coordsValue}>
                  {selectedLocation.latitude.toFixed(6)}, {selectedLocation.longitude.toFixed(6)}
                </Text>
              </View>
            )}

            <View style={styles.modalButtons}>
              <TouchableOpacity
                style={styles.addButton}
                onPress={handleAddZone}
              >
                <Text style={styles.buttonText}>Add Zone</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.cancelButton}
                onPress={() => {
                  setShowAddModal(false);
                  setZoneName('');
                  setZoneCode('');
                  setZoneType('timed');
                  setDefaultTimeLimit(120);
                  setSelectedLocation(null);
                }}
              >
                <Text style={styles.buttonText}>Cancel</Text>
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
    top: Platform.OS === 'web' ? 10 : 50,
    right: 10,
  },
  locationButton: {
    backgroundColor: '#007AFF',
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
    elevation: 5,
  },
  buttonIcon: {
    fontSize: 20,
  },
  instructionsOverlay: {
    position: 'absolute',
    bottom: 10,
    left: 10,
    right: 10,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    padding: 10,
    borderRadius: 8,
    alignItems: 'center',
  },
  instructionsText: {
    color: '#FFF',
    fontSize: 14,
  },
  zoneListContainer: {
    maxHeight: 280,
    backgroundColor: '#1a1a1a',
    borderTopWidth: 1,
    borderTopColor: '#333',
  },
  zoneListHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  zoneListTitle: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  headerButtons: {
    flexDirection: 'row',
    gap: 8,
  },
  seedButton: {
    backgroundColor: '#4CAF50',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 6,
  },
  seedButtonText: {
    color: '#FFF',
    fontSize: 12,
    fontWeight: '600',
  },
  clearAllButton: {
    backgroundColor: '#FF3B30',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 6,
  },
  clearAllButtonText: {
    color: '#FFF',
    fontSize: 12,
    fontWeight: '600',
  },
  zoneList: {
    padding: 12,
  },
  zoneCard: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: '#000',
    padding: 12,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#333',
    marginBottom: 8,
  },
  zoneInfo: {
    flex: 1,
  },
  zoneName: {
    color: '#FFF',
    fontSize: 14,
    fontWeight: 'bold',
  },
  zoneMetaRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginTop: 4,
  },
  zoneCode: {
    color: '#007AFF',
    fontSize: 12,
    fontWeight: '600',
  },
  zoneType: {
    fontSize: 10,
    fontWeight: '600',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
  },
  zoneTypePermit: {
    backgroundColor: '#007AFF',
    color: '#FFF',
  },
  zoneTypeTimed: {
    backgroundColor: '#4CAF50',
    color: '#FFF',
  },
  zoneTimeLimit: {
    color: '#888',
    fontSize: 10,
  },
  zoneCoords: {
    color: '#666',
    fontSize: 10,
    marginTop: 4,
  },
  deleteButton: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#FF3B3020',
    justifyContent: 'center',
    alignItems: 'center',
  },
  deleteButtonText: {
    color: '#FF3B30',
    fontSize: 20,
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
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFF',
    marginBottom: 20,
    textAlign: 'center',
  },
  inputGroup: {
    marginBottom: 16,
  },
  inputLabel: {
    color: '#888',
    fontSize: 12,
    fontWeight: '600',
    marginBottom: 6,
  },
  input: {
    backgroundColor: '#000',
    borderWidth: 1,
    borderColor: '#333',
    borderRadius: 8,
    padding: 12,
    color: '#FFF',
    fontSize: 14,
  },
  zoneTypeToggle: {
    flexDirection: 'row',
    gap: 12,
  },
  zoneTypeButton: {
    flex: 1,
    paddingVertical: 10,
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
  timeLimitOptions: {
    flexDirection: 'row',
    gap: 8,
  },
  timeLimitButton: {
    flex: 1,
    paddingVertical: 8,
    borderRadius: 6,
    borderWidth: 1,
    borderColor: '#333',
    alignItems: 'center',
    backgroundColor: '#000',
  },
  timeLimitButtonActive: {
    backgroundColor: '#4CAF50',
    borderColor: '#4CAF50',
  },
  timeLimitButtonText: {
    color: '#888',
    fontSize: 12,
    fontWeight: '600',
  },
  timeLimitButtonTextActive: {
    color: '#FFF',
  },
  coordsDisplay: {
    backgroundColor: '#000',
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
  },
  coordsLabel: {
    color: '#888',
    fontSize: 12,
    fontWeight: '600',
    marginBottom: 4,
  },
  coordsValue: {
    color: '#4CAF50',
    fontSize: 12,
    fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace',
  },
  modalButtons: {
    gap: 12,
  },
  addButton: {
    backgroundColor: '#4CAF50',
    padding: 14,
    borderRadius: 8,
    alignItems: 'center',
  },
  cancelButton: {
    backgroundColor: '#333',
    padding: 14,
    borderRadius: 8,
    alignItems: 'center',
  },
  buttonText: {
    color: '#FFF',
    fontSize: 14,
    fontWeight: '600',
  },
});