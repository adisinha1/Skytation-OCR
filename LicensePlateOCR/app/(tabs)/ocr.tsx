import React, { useState, useRef } from 'react';
import { View, Text, StyleSheet, ActivityIndicator, TouchableOpacity, ScrollView } from 'react-native';
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
  const [isProcessing, setIsProcessing] = useState(false);
  const [captureCount, setCaptureCount] = useState(0);
  const [lastQuality, setLastQuality] = useState('');
  const [classification, setClassification] = useState<Classification | null>(null);
  const [rawText, setRawText] = useState('');
  const [confidence, setConfidence] = useState(0);

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

      // Wait 500ms for camera to focus
      await new Promise(resolve => setTimeout(resolve, 500));

      // Capture at maximum quality
      const photo = await cameraRef.current.takePictureAsync({
        quality: 1.0,
        base64: true,
      });

      if (!photo?.base64) {
        throw new Error('Failed to capture photo');
      }

      setRawText('Sending to server...');

      // Send to backend
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
      {/* Camera */}
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

      {/* Results Panel */}
      <ScrollView style={styles.resultsPanel}>
        {/* Quality Status */}
        {lastQuality && (
          <Text style={styles.qualityText}>{lastQuality}</Text>
        )}

        {/* Classification Results */}
        {classification && (
          <View style={styles.classificationContainer}>
            {/* State Info */}
            {classification.state && (
              <View style={styles.infoBlock}>
                <Text style={styles.blockLabel}>STATE</Text>
                <View style={styles.blockContent}>
                  <Text style={styles.blockText}>{classification.state}</Text>
                  <Text style={styles.blockSubtext}>{classification.state_abbreviation}</Text>
                </View>
              </View>
            )}

            {/* License Number */}
            {classification.license_number && (
              <View style={styles.infoBlock}>
                <Text style={styles.blockLabel}>LICENSE PLATE</Text>
                <Text style={styles.licensePlateNumber}>{classification.license_number}</Text>
              </View>
            )}

            {/* Expiration Date */}
            {classification.expiration_date && (
              <View style={styles.infoBlock}>
                <Text style={styles.blockLabel}>EXPIRATION</Text>
                <Text style={styles.expirationText}>{classification.expiration_date}</Text>
              </View>
            )}

            {/* State Slogan */}
            {classification.slogan && (
              <View style={styles.infoBlock}>
                <Text style={styles.blockLabel}>STATE MOTTO</Text>
                <Text style={styles.sloganText}>{classification.slogan}</Text>
              </View>
            )}

            {/* Other Text */}
            {classification.other_text && classification.other_text.length > 0 && (
              <View style={styles.infoBlock}>
                <Text style={styles.blockLabel}>OTHER TEXT</Text>
                <Text style={styles.otherText}>{classification.other_text.join(' ')}</Text>
              </View>
            )}
          </View>
        )}

        {/* Raw Text and Confidence */}
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

        {/* Capture Stats */}
        <Text style={styles.statsText}>Captures: {captureCount}</Text>
      </ScrollView>

      {/* Capture Button */}
      <View style={styles.buttonContainer}>
        <TouchableOpacity
          style={[styles.button, isProcessing && styles.buttonDisabled]}
          onPress={handleTakePhoto}
          disabled={isProcessing}
        >
          <Text style={styles.buttonText}>
            {isProcessing ? 'Processing...' : 'Capture & Analyze'}
          </Text>
        </TouchableOpacity>
      </View>

      {/* Loading Overlay */}
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
  resultsPanel: {
    flex: 0.45,
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