import React, { useState, useRef } from 'react';
import { View, Text, StyleSheet, ActivityIndicator, TouchableOpacity, ScrollView, Image } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';

interface Classification {
  state: string | null;
  state_abbreviation: string | null;
  license_number: string | null;
  expiration_date: string | null;
  slogan: string | null;
  other_text: string[];
}

export default function OCRScreen() {
  const cameraRef = useRef<CameraView>(null);
  const [permission, requestPermission] = useCameraPermissions();
  
  // State declarations
  const [isProcessing, setIsProcessing] = useState(false);
  const [captureCount, setCaptureCount] = useState(0);
  const [lastQuality, setLastQuality] = useState('');
  const [classification, setClassification] = useState<Classification | null>(null);
  const [rawText, setRawText] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [lastPhoto, setLastPhoto] = useState<string | null>(null);
  const [debugImages, setDebugImages] = useState<Array<{name: string, data: string}>>([]);
  const [currentDebugImageIndex, setCurrentDebugImageIndex] = useState(0);

  const BACKEND_URL = 'http://10.0.0.38:5001';

  React.useEffect(() => {
    if (!permission?.granted) {
      requestPermission();
    }
  }, [permission]);

  const handleTakePhoto = async () => {
    if (!cameraRef.current || isProcessing) return;

    try {
      setIsProcessing(true);
      setClassification(null);
      setRawText('Focusing...');

      await new Promise(resolve => setTimeout(resolve, 500));

      const photo = await cameraRef.current.takePictureAsync({
        quality: 1.0,
        base64: true,
      });

      if (!photo?.base64) {
        throw new Error('Failed to capture photo');
      }

      setLastPhoto(`data:image/jpg;base64,${photo.base64}`);
      setRawText('Sending to server...');

      const response = await fetch(`${BACKEND_URL}/process-frame`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame: photo.base64 }),
      });

      if (response.ok) {
        const data = await response.json();
        setRawText(data.text || 'No text detected');
        setConfidence(data.confidence || 0);
        setClassification(data.classification || null);
        setLastQuality(data.quality_status || '');
        setDebugImages(data.debug_images || []);
        setCurrentDebugImageIndex(0);
        setCaptureCount(c => c + 1);
      } else {
        setRawText('Backend error');
      }

      setIsProcessing(false);
    } catch (err) {
      setRawText('Error: ' + String(err));
      setIsProcessing(false);
    }
  };

  const handleCaptureNewPlate = () => {
    setLastPhoto(null);
    setRawText('');
    setConfidence(0);
    setClassification(null);
    setLastQuality('');
    setDebugImages([]);
    setCurrentDebugImageIndex(0);
  };

  if (!permission) {
    return (
      <View style={styles.container}>
        <Text style={styles.text}>Requesting permission...</Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.text}>Camera permission denied</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Camera or Debug Section */}
      {!lastPhoto ? (
        <CameraView ref={cameraRef} style={styles.camera} facing="back">
          <View style={styles.cameraOverlay}>
            <View style={styles.focusGuide}>
              <Text style={styles.focusText}>Position license plate here</Text>
            </View>

            <View style={styles.guidanceContainer}>
              <Text style={styles.guidanceText}>✓ Good lighting</Text>
              <Text style={styles.guidanceText}>✓ Straight angle</Text>
              <Text style={styles.guidanceText}>✓ 1-2 feet away</Text>
              <Text style={styles.guidanceText}>✓ Keep steady</Text>
            </View>
          </View>
        </CameraView>
      ) : (
        <View style={styles.debugSection}>
          {debugImages.length > 0 ? (
            <>
              <View style={styles.debugImageInfo}>
                <Text style={styles.debugImageLabel}>
                  {debugImages[currentDebugImageIndex]?.name}
                </Text>
                <Text style={styles.debugImageCounter}>
                  {currentDebugImageIndex + 1} of {debugImages.length}
                </Text>
              </View>
              
              <View style={styles.debugImageDisplay}>
                <Image
                  source={{ uri: `data:image/jpg;base64,${debugImages[currentDebugImageIndex]?.data}` }}
                  style={styles.debugImage}
                />
              </View>
              
              <View style={styles.debugImageNav}>
                <TouchableOpacity
                  onPress={() => setCurrentDebugImageIndex(Math.max(0, currentDebugImageIndex - 1))}
                  disabled={currentDebugImageIndex === 0}
                >
                  <Text style={styles.navButton}>← Prev</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  onPress={() => setCurrentDebugImageIndex(Math.min(debugImages.length - 1, currentDebugImageIndex + 1))}
                  disabled={currentDebugImageIndex === debugImages.length - 1}
                >
                  <Text style={styles.navButton}>Next →</Text>
                </TouchableOpacity>
              </View>
            </>
          ) : (
            <Text style={styles.debugLabel}>Processing...</Text>
          )}
        </View>
      )}

      {/* Results Panel */}
      <ScrollView style={styles.resultsPanel}>
        {lastQuality && (
          <Text style={styles.qualityText}>{lastQuality}</Text>
        )}

        {classification && (
          <View style={styles.classificationContainer}>
            {classification.state && (
              <View style={styles.infoBlock}>
                <Text style={styles.blockLabel}>STATE</Text>
                <View style={styles.blockContent}>
                  <Text style={styles.blockText}>{classification.state}</Text>
                  <Text style={styles.blockSubtext}>{classification.state_abbreviation}</Text>
                </View>
              </View>
            )}

            {classification.license_number && (
              <View style={styles.infoBlock}>
                <Text style={styles.blockLabel}>LICENSE PLATE</Text>
                <Text style={styles.licensePlateNumber}>{classification.license_number}</Text>
              </View>
            )}

            {classification.expiration_date && (
              <View style={styles.infoBlock}>
                <Text style={styles.blockLabel}>EXPIRATION</Text>
                <Text style={styles.expirationText}>{classification.expiration_date}</Text>
              </View>
            )}

            {classification.slogan && (
              <View style={styles.infoBlock}>
                <Text style={styles.blockLabel}>STATE MOTTO</Text>
                <Text style={styles.sloganText}>{classification.slogan}</Text>
              </View>
            )}

            {classification.other_text && classification.other_text.length > 0 && (
              <View style={styles.infoBlock}>
                <Text style={styles.blockLabel}>OTHER TEXT</Text>
                <Text style={styles.otherText}>{classification.other_text.join(' ')}</Text>
              </View>
            )}
          </View>
        )}

        <View style={styles.rawDataBlock}>
          <Text style={styles.blockLabel}>RAW TEXT</Text>
          <Text style={styles.rawText} numberOfLines={4}>{rawText}</Text>
          {confidence > 0 && (
            <View style={styles.confidenceSection}>
              <Text style={styles.confidenceLabel}>Confidence: {(confidence * 100).toFixed(1)}%</Text>
              <View style={styles.progressBar}>
                <View
                  style={[
                    styles.progressFill,
                    {
                      width: `${confidence * 100}%`,
                      backgroundColor: confidence > 0.8 ? '#4CAF50' : confidence > 0.6 ? '#FFA500' : '#F44336',
                    },
                  ]}
                />
              </View>
            </View>
          )}
        </View>

        <Text style={styles.statsText}>Captures: {captureCount}</Text>
      </ScrollView>

      {/* Buttons */}
      <View style={styles.buttonContainer}>
        {!lastPhoto ? (
          <TouchableOpacity
            style={[styles.button, isProcessing && styles.buttonDisabled]}
            onPress={handleTakePhoto}
            disabled={isProcessing}
          >
            <Text style={styles.buttonText}>
              {isProcessing ? 'Processing...' : '📸 Capture & Analyze'}
            </Text>
          </TouchableOpacity>
        ) : (
          <TouchableOpacity
            style={styles.button}
            onPress={handleCaptureNewPlate}
          >
            <Text style={styles.buttonText}>📷 Capture New Plate</Text>
          </TouchableOpacity>
        )}
      </View>

      {isProcessing && (
        <View style={styles.loadingOverlay}>
          <ActivityIndicator size="large" color="#FFF" />
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  text: {
    color: '#FFF',
    fontSize: 18,
  },
  camera: {
    flex: 0.5,
    width: '100%',
    backgroundColor: '#000',
  },
  cameraOverlay: {
    flex: 1,
    justifyContent: 'space-around',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.2)',
    paddingVertical: 20,
  },
  focusGuide: {
    width: '85%',
    height: 100,
    borderWidth: 3,
    borderColor: 'rgba(0, 255, 0, 0.7)',
    borderRadius: 8,
    justifyContent: 'center',
    alignItems: 'center',
  },
  focusText: {
    color: 'rgba(0, 255, 0, 0.9)',
    fontSize: 13,
    fontWeight: 'bold',
  },
  guidanceContainer: {
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 8,
    width: '85%',
  },
  guidanceText: {
    color: '#AAA',
    fontSize: 11,
    marginVertical: 3,
  },
  debugSection: {
    flex: 0.5,
    backgroundColor: '#1a1a1a',
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 12,
  },
  debugLabel: {
    color: '#00FF00',
    fontSize: 14,
    fontWeight: 'bold',
  },
  debugImageInfo: {
    alignItems: 'center',
    marginBottom: 8,
  },
  debugImageLabel: {
    color: '#00FF00',
    fontSize: 13,
    fontWeight: 'bold',
  },
  debugImageCounter: {
    color: '#888',
    fontSize: 11,
    marginTop: 4,
  },
  debugImageDisplay: {
    flex: 1,
    width: '100%',
    backgroundColor: '#000',
    borderRadius: 6,
    borderWidth: 2,
    borderColor: '#00FF00',
    marginVertical: 8,
    justifyContent: 'center',
    alignItems: 'center',
    overflow: 'hidden',
  },
  debugImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'contain',
  },
  debugImageNav: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    width: '100%',
    paddingHorizontal: 20,
  },
  navButton: {
    color: '#00FF00',
    fontSize: 12,
    fontWeight: 'bold',
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  resultsPanel: {
    flex: 0.4,
    backgroundColor: '#1a1a1a',
    padding: 14,
  },
  qualityText: {
    color: '#FFA500',
    fontSize: 11,
    marginBottom: 12,
    fontStyle: 'italic',
    textAlign: 'center',
  },
  classificationContainer: {
    marginBottom: 14,
  },
  infoBlock: {
    backgroundColor: '#000',
    borderRadius: 6,
    padding: 10,
    marginBottom: 8,
    borderLeftWidth: 3,
    borderLeftColor: '#00FF00',
  },
  blockLabel: {
    color: '#888',
    fontSize: 9,
    fontWeight: '700',
    textTransform: 'uppercase',
    marginBottom: 4,
    letterSpacing: 0.5,
  },
  blockContent: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  blockText: {
    color: '#00FF00',
    fontSize: 14,
    fontWeight: 'bold',
  },
  blockSubtext: {
    color: '#00DD00',
    fontSize: 12,
    fontWeight: '600',
  },
  licensePlateNumber: {
    color: '#FFD700',
    fontSize: 16,
    fontWeight: 'bold',
    fontFamily: 'monospace',
    letterSpacing: 2,
  },
  expirationText: {
    color: '#FF6B9D',
    fontSize: 14,
    fontWeight: 'bold',
  },
  sloganText: {
    color: '#87CEEB',
    fontSize: 13,
    fontStyle: 'italic',
    fontWeight: '600',
  },
  otherText: {
    color: '#CCC',
    fontSize: 12,
  },
  rawDataBlock: {
    backgroundColor: '#000',
    borderRadius: 6,
    padding: 10,
    marginBottom: 10,
    borderLeftWidth: 3,
    borderLeftColor: '#666',
  },
  rawText: {
    color: '#AAA',
    fontSize: 11,
    fontFamily: 'monospace',
    marginBottom: 6,
  },
  confidenceSection: {
    marginTop: 6,
  },
  confidenceLabel: {
    color: '#888',
    fontSize: 10,
    marginBottom: 4,
  },
  progressBar: {
    height: 6,
    backgroundColor: '#333',
    borderRadius: 3,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
  },
  statsText: {
    color: '#666',
    fontSize: 10,
    textAlign: 'center',
  },
  buttonContainer: {
    backgroundColor: '#000',
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderTopWidth: 1,
    borderTopColor: '#333',
  },
  button: {
    backgroundColor: '#007AFF',
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  buttonDisabled: {
    backgroundColor: '#555',
  },
  buttonText: {
    color: '#FFF',
    fontSize: 15,
    fontWeight: '600',
  },
  loadingOverlay: {
    ...StyleSheet.absoluteFill,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    justifyContent: 'center',
    alignItems: 'center',
  },
});