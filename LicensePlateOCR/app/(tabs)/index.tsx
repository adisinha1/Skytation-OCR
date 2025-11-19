import { Image } from 'expo-image';
import { StyleSheet, ScrollView, TouchableOpacity, View } from 'react-native';
import { useState, useCallback } from 'react';
import { useFocusEffect } from 'expo-router';

import ParallaxScrollView from '@/components/parallax-scroll-view';
import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { getScans, clearScans, ScanRecord } from '@/app/scanStorage';
import { getZones, findZoneByCoordinates, CampusZone } from '@/app/campusZones';

export default function HomeScreen() {
  const [scans, setScans] = useState<ScanRecord[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [campusZones, setCampusZones] = useState<CampusZone[]>([]);

  const loadScans = async () => {
    setIsLoading(true);
    const scanHistory = await getScans();
    const zones = await getZones();
    setScans(scanHistory);
    setCampusZones(zones);
    setIsLoading(false);
  };

  useFocusEffect(
    useCallback(() => {
      loadScans();
    }, [])
  );

  const handleClearHistory = async () => {
    await clearScans();
    setScans([]);
  };

  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    });
  };

  const getCampusLocation = (gps: {latitude: number, longitude: number} | null | undefined) => {
    if (!gps) return null;

    const zone = findZoneByCoordinates(gps.latitude, gps.longitude, campusZones);
    
    if (zone) {
      return {
        name: zone.name,
        zone: zone.code,
      };
    }

    return null;
  };

  return (
    <ParallaxScrollView
      headerBackgroundColor={{ light: '#A1CEDC', dark: '#1D3D47' }}
      headerImage={
        <Image
          source={require('@/assets/images/partial-react-logo.png')}
          style={styles.reactLogo}
        />
      }>
      <ThemedView style={styles.titleContainer}>
        <ThemedText type="title">Scan History</ThemedText>
      </ThemedView>

      <ThemedView style={styles.headerRow}>
        <ThemedText type="subtitle">Recent Scans</ThemedText>
        {scans.length > 0 && (
          <TouchableOpacity onPress={handleClearHistory} style={styles.clearButton}>
            <ThemedText style={styles.clearButtonText}>Clear All</ThemedText>
          </TouchableOpacity>
        )}
      </ThemedView>

      {isLoading ? (
        <ThemedView style={styles.emptyContainer}>
          <ThemedText>Loading...</ThemedText>
        </ThemedView>
      ) : scans.length === 0 ? (
        <ThemedView style={styles.emptyContainer}>
          <ThemedText style={styles.emptyText}>No scans yet</ThemedText>
          <ThemedText style={styles.emptySubtext}>
            Go to the OCR tab to scan your first license plate
          </ThemedText>
        </ThemedView>
      ) : (
        <ScrollView style={styles.scanList}>
          {scans.map((scan) => {
            const campusLocation = getCampusLocation(scan.gpsCoordinates);
            
            return (
              <ThemedView key={scan.id} style={styles.scanCard}>
                <Image source={{ uri: scan.image }} style={styles.scanImage} />
                
                <View style={styles.scanInfo}>
                  {/* Top Row: Plate, State, Time */}
                  <View style={styles.scanDetails}>
                    {scan.licenseNumber && (
                      <ThemedText style={styles.licenseNumber}>
                        {scan.licenseNumber}
                      </ThemedText>
                    )}
                    
                    {scan.stateAbbreviation && (
                      <View style={styles.stateBadge}>
                        <ThemedText style={styles.stateText}>
                          {scan.stateAbbreviation}
                        </ThemedText>
                      </View>
                    )}

                    <View style={styles.timeBadge}>
                      <ThemedText style={styles.timeText}>
                        {formatTime(scan.timestamp)}
                      </ThemedText>
                    </View>
                  </View>

                  {/* Campus Location - Zone Code + Name on same line */}
                  {campusLocation && (
                    <View style={styles.locationRow}>
                      <View style={styles.zoneBadge}>
                        <ThemedText style={styles.zoneText}>
                          {campusLocation.zone}
                        </ThemedText>
                      </View>
                      <ThemedText style={styles.locationName}>
                        📍 {campusLocation.name}
                      </ThemedText>
                    </View>
                  )}
                </View>
              </ThemedView>
            );
          })}
        </ScrollView>
      )}
    </ParallaxScrollView>
  );
}

const styles = StyleSheet.create({
  titleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 16,
  },
  headerRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  clearButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: '#FF3B30',
    borderRadius: 6,
  },
  clearButtonText: {
    color: '#FFF',
    fontSize: 12,
    fontWeight: '600',
  },
  reactLogo: {
    height: 178,
    width: 290,
    bottom: 0,
    left: 0,
    position: 'absolute',
  },
  emptyContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 60,
    gap: 8,
  },
  emptyText: {
    fontSize: 18,
    fontWeight: '600',
    opacity: 0.6,
  },
  emptySubtext: {
    fontSize: 14,
    opacity: 0.4,
    textAlign: 'center',
    paddingHorizontal: 40,
  },
  scanList: {
    flex: 1,
  },
  scanCard: {
    flexDirection: 'row',
    backgroundColor: '#000',
    borderRadius: 12,
    marginBottom: 12,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: '#333',
  },
  scanImage: {
    width: 120,
    height: 90,
    backgroundColor: '#1a1a1a',
  },
  scanInfo: {
    flex: 1,
    padding: 12,
    justifyContent: 'space-between',
  },
  scanDetails: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    flexWrap: 'wrap',
  },
  licenseNumber: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FFD700',
    fontFamily: 'monospace',
    letterSpacing: 1,
  },
  stateBadge: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  stateText: {
    color: '#FFF',
    fontSize: 12,
    fontWeight: '700',
  },
  timeBadge: {
    backgroundColor: '#4CAF50',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  timeText: {
    color: '#FFF',
    fontSize: 11,
    fontWeight: '600',
  },
  locationRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginTop: 4,
  },
  zoneBadge: {
    backgroundColor: '#FF6B35',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 4,
  },
  zoneText: {
    color: '#FFF',
    fontSize: 12,
    fontWeight: 'bold',
    letterSpacing: 1,
  },
  locationName: {
    fontSize: 12,
    fontWeight: '600',
    color: '#AAA',
    flex: 1,
  },
});
