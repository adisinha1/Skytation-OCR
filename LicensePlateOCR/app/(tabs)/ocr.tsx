import React, { useState, useRef, useEffect } from 'react';
import { View, Text, StyleSheet, ActivityIndicator, TouchableOpacity, ScrollView, Image, Modal, Dimensions, Platform, Alert } from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { saveScan } from '@/app/scanStorage';
import * as Location from 'expo-location';

interface Classification {
  state: string | null;
  state_abbreviation: string | null;
  license_number: string | null;
  expiration_date: string | null;
  slogan: string | null;
  other_text: string[];
}

interface LastCapture {
  photo: string;
  classification: Classification | null;
  rawText: string;
  confidence: number;
  quality: string;
  debugImages: Array<{name: string, data: string}>;
  source: 'phone' | 'drone';
  timestamp: Date;
}

interface EnforcementResult {
  result: 'approved' | 'violation' | 'unknown';
  reason: string;
  message: string;
  detected_zone?: string;
  zone_type?: 'permit' | 'timed';
  dwell_minutes?: number;
  limit_minutes?: number;
}

export default function OCRScreen() {
  const cameraRef = useRef<CameraView>(null);
  const [permission, requestPermission] = useCameraPermissions();
  const [locationPermission, setLocationPermission] = useState<Location.PermissionStatus | null>(null);
  
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
  const [captureSource, setCaptureSource] = useState<'phone' | 'drone'>('phone');
  const [enforcementResult, setEnforcementResult] = useState<EnforcementResult | null>(null);
  
  // New state for last capture modal
  const [lastCapture, setLastCapture] = useState<LastCapture | null>(null);
  const [showLastCaptureModal, setShowLastCaptureModal] = useState(false);
  const [lastCaptureDebugIndex, setLastCaptureDebugIndex] = useState(0);

  const BACKEND_URL = 'http://10.0.0.66:5001';
  const ENFORCEMENT_URL = 'http://10.0.0.66:8000';

  // Request location permissions on mount
  useEffect(() => {
    (async () => {
      const { status } = await Location.requestForegroundPermissionsAsync();
      setLocationPermission(status);
    })();
  }, []);

  React.useEffect(() => {
    if (!permission?.granted) {
      requestPermission();
    }
  }, [permission]);

  const getCurrentLocation = async () => {
    try {
      // Check if we already have permission
      if (locationPermission !== 'granted') {
        console.log('Location permission not granted');
        return null;
      }

      const location = await Location.getCurrentPositionAsync({
        accuracy: Location.Accuracy.High,
      });
      
      return {
        latitude: location.coords.latitude,
        longitude: location.coords.longitude,
      };
    } catch (error) {
      console.error('Error getting location:', error);
      return null;
    }
  };
  
  const logToEnforcement = async (
    plateNumber: string, 
    state: string | null, 
    confidence: number, 
    source: 'phone' | 'drone',
    imageData?: string
  ) => {
      try {
        // Get current GPS location
        const location = await getCurrentLocation();
        
        const response = await fetch(`${ENFORCEMENT_URL}/api/ocr_event`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            plate_text: plateNumber,
            confidence: confidence,
            timestamp: new Date().toISOString(),
            location: 'timed', // Will be auto-detected by backend based on GPS
            state: state,
            source: source,
            image_data: imageData,
            latitude: location?.latitude,
            longitude: location?.longitude,
          }),
        });

        if (response.ok) {
          const result = await response.json();
          setEnforcementResult(result);
          
          // Show alert for violations
          if (result.result === 'violation') {
            Alert.alert(
              '⚠️ VIOLATION DETECTED',
              result.message,
              [{ text: 'OK', style: 'destructive' }]
            );
          }
          
          return result;
        }
      } catch (error) {
        console.error('Error logging to enforcement:', error);
      }
      return null;
    };

  const saveToLastCapture = (photo: string, classificationData: Classification | null, text: string, conf: number, quality: string, debugImgs: Array<{name: string, data: string}>, source: 'phone' | 'drone') => {
    setLastCapture({
      photo,
      classification: classificationData,
      rawText: text,
      confidence: conf,
      quality,
      debugImages: debugImgs,
      source,
      timestamp: new Date(),
    });
  };

  const handleTakePhoto = async () => {
    if (!cameraRef.current || isProcessing) return;

    try {
      setIsProcessing(true);
      setClassification(null);
      setRawText('Focusing...');
      setCaptureSource('phone');
      setEnforcementResult(null);

      await new Promise(resolve => setTimeout(resolve, 500));

      const photo = await cameraRef.current.takePictureAsync({
        quality: 1.0,
        base64: true,
      });

      if (!photo?.base64) {
        throw new Error('Failed to capture photo');
      }

      const photoUri = `data:image/jpg;base64,${photo.base64}`;
      setLastPhoto(photoUri);
      setRawText('Sending to server...');

      const response = await fetch(`${BACKEND_URL}/process-frame`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame: photo.base64 }),
      });

      if (response.ok) {
        const data = await response.json();
        const text = data.text || 'No text detected';
        const conf = data.confidence || 0;
        const classData = data.classification || null;
        const quality = data.quality_status || '';
        const debugImgs = data.debug_images || [];
        
        setRawText(text);
        setConfidence(conf);
        setClassification(classData);
        setLastQuality(quality);
        setDebugImages(debugImgs);
        setCurrentDebugImageIndex(0);
        setCaptureCount(c => c + 1);
        
        // Save to last capture
        saveToLastCapture(photoUri, classData, text, conf, quality, debugImgs, 'phone');
        
        if (classData?.license_number && photo?.base64) {
          await saveScan({
            licenseNumber: classData.license_number,
            stateAbbreviation: classData.state_abbreviation,
            image: photoUri,
          });
          
          await logToEnforcement(
            classData.license_number,
            classData.state_abbreviation,
            conf,
            'phone',
            photo.base64 
          );
        }
      } else {
        setRawText('Backend error');
      }

      setIsProcessing(false);
    } catch (err) {
      setRawText('Error: ' + String(err));
      setIsProcessing(false);
    }
  };

  const handleCaptureDrone = async () => {
    if (isProcessing) return;

    try {
      setIsProcessing(true);
      setClassification(null);
      setRawText('Capturing from drone...');
      setCaptureSource('drone');
      setLastPhoto(null);
      setEnforcementResult(null);

      const response = await fetch(`${BACKEND_URL}/capture-drone`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });

      if (response.ok) {
        const data = await response.json();
        
        const photoUri = data.captured_image || null;
        if (photoUri) {
          setLastPhoto(photoUri);
        }
        
        const text = data.text || 'No text detected';
        const conf = data.confidence || 0;
        const classData = data.classification || null;
        const quality = data.quality_status || `Frame: ${data.frame_width}x${data.frame_height}`;
        const debugImgs = data.debug_images || [];
        
        setRawText(text);
        setConfidence(conf);
        setClassification(classData);
        setLastQuality(quality);
        setDebugImages(debugImgs);
        setCurrentDebugImageIndex(0);
        setCaptureCount(c => c + 1);
        
        // Save to last capture
        if (photoUri) {
          saveToLastCapture(photoUri, classData, text, conf, quality, debugImgs, 'drone');
        }
        
        if (classData?.license_number && photoUri) {
          await saveScan({
            licenseNumber: classData.license_number,
            stateAbbreviation: classData.state_abbreviation,
            image: photoUri,
          });
          
          await logToEnforcement(
            classData.license_number,
            classData.state_abbreviation,
            conf,
            'drone',
            data.captured_image?.replace('data:image/jpg;base64,', '')
          );
        }
      } else {
        const errorData = await response.json();
        setRawText(`Drone capture error: ${errorData.error || 'Unknown error'}`);
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
    setEnforcementResult(null);
  };

  const renderLastCaptureModal = () => (
    <Modal
      visible={showLastCaptureModal}
      animationType="slide"
      transparent={false}
      onRequestClose={() => setShowLastCaptureModal(false)}
    >
      <View style={styles.modalContainer}>
        <View style={styles.modalHeader}>
          <Text style={styles.modalTitle}>Last Capture</Text>
          <TouchableOpacity onPress={() => setShowLastCaptureModal(false)}>
            <Text style={styles.closeButton}>✕</Text>
          </TouchableOpacity>
        </View>
        
        {lastCapture ? (
          <ScrollView style={styles.modalContent}>
            <Text style={styles.timestampText}>
              {lastCapture.timestamp.toLocaleString()} • {lastCapture.source === 'drone' ? '🚁 Drone' : '📸 Phone'}
            </Text>
            
            {lastCapture.debugImages.length > 0 ? (
              <View style={styles.modalDebugSection}>
                <View style={styles.debugImageInfo}>
                  <Text style={styles.debugImageLabel}>
                    {lastCapture.debugImages[lastCaptureDebugIndex]?.name}
                  </Text>
                  <Text style={styles.debugImageCounter}>
                    {lastCaptureDebugIndex + 1} of {lastCapture.debugImages.length}
                  </Text>
                </View>
                
                <View style={styles.modalImageContainer}>
                  <Image
                    source={{ uri: `data:image/jpg;base64,${lastCapture.debugImages[lastCaptureDebugIndex]?.data}` }}
                    style={styles.modalImage}
                  />
                </View>
                
                <View style={styles.debugImageNav}>
                  <TouchableOpacity
                    onPress={() => setLastCaptureDebugIndex(Math.max(0, lastCaptureDebugIndex - 1))}
                    disabled={lastCaptureDebugIndex === 0}
                  >
                    <Text style={[styles.navButton, lastCaptureDebugIndex === 0 && styles.navButtonDisabled]}>← Prev</Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    onPress={() => setLastCaptureDebugIndex(Math.min(lastCapture.debugImages.length - 1, lastCaptureDebugIndex + 1))}
                    disabled={lastCaptureDebugIndex === lastCapture.debugImages.length - 1}
                  >
                    <Text style={[styles.navButton, lastCaptureDebugIndex === lastCapture.debugImages.length - 1 && styles.navButtonDisabled]}>Next →</Text>
                  </TouchableOpacity>
                </View>
              </View>
            ) : (
              <View style={styles.modalImageContainer}>
                <Image
                  source={{ uri: lastCapture.photo }}
                  style={styles.modalImage}
                />
              </View>
            )}
            
            {lastCapture.classification && (
              <View style={styles.classificationContainer}>
                {lastCapture.classification.state && (
                  <View style={styles.infoBlock}>
                    <Text style={styles.blockLabel}>STATE</Text>
                    <View style={styles.blockContent}>
                      <Text style={styles.blockText}>{lastCapture.classification.state}</Text>
                      <Text style={styles.blockSubtext}>{lastCapture.classification.state_abbreviation}</Text>
                    </View>
                  </View>
                )}

                {lastCapture.classification.license_number && (
                  <View style={styles.infoBlock}>
                    <Text style={styles.blockLabel}>LICENSE PLATE</Text>
                    <Text style={styles.licensePlateNumber}>{lastCapture.classification.license_number}</Text>
                  </View>
                )}

                {lastCapture.classification.expiration_date && (
                  <View style={styles.infoBlock}>
                    <Text style={styles.blockLabel}>EXPIRATION</Text>
                    <Text style={styles.expirationText}>{lastCapture.classification.expiration_date}</Text>
                  </View>
                )}

                {lastCapture.classification.slogan && (
                  <View style={styles.infoBlock}>
                    <Text style={styles.blockLabel}>STATE MOTTO</Text>
                    <Text style={styles.sloganText}>{lastCapture.classification.slogan}</Text>
                  </View>
                )}
              </View>
            )}
            
            <View style={styles.rawDataBlock}>
              <Text style={styles.blockLabel}>RAW TEXT</Text>
              <Text style={styles.rawText}>{lastCapture.rawText}</Text>
              {lastCapture.confidence > 0 && (
                <View style={styles.confidenceSection}>
                  <Text style={styles.confidenceLabel}>Confidence: {(lastCapture.confidence * 100).toFixed(1)}%</Text>
                  <View style={styles.progressBar}>
                    <View
                      style={[
                        styles.progressFill,
                        {
                          width: `${lastCapture.confidence * 100}%`,
                          backgroundColor: lastCapture.confidence > 0.8 ? '#4CAF50' : lastCapture.confidence > 0.6 ? '#FFA500' : '#F44336',
                        },
                      ]}
                    />
                  </View>
                </View>
              )}
            </View>
          </ScrollView>
        ) : (
          <View style={styles.noCaptureContainer}>
            <Text style={styles.noCaptureText}>No previous capture available</Text>
          </View>
        )}
      </View>
    </Modal>
  );

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
        <TouchableOpacity
          style={[styles.button, styles.droneButton, { marginTop: 20 }]}
          onPress={handleCaptureDrone}
          disabled={isProcessing}
        >
          <Text style={styles.buttonText}>
            {isProcessing ? 'Capturing...' : '🚁 Capture from Drone'}
          </Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      {/* Camera or Debug Section */}
      {!lastPhoto ? (
        <CameraView ref={cameraRef} style={styles.camera} facing="back">
          <View style={styles.cameraOverlay}>
            <View style={styles.focusGuide} />
          </View>
        </CameraView>
      ) : (
        <View style={styles.debugSection}>
          {debugImages.length > 0 ? (
            <>
              <View style={styles.debugImageInfo}>
                <Text style={styles.debugImageLabel}>
                  {debugImages[currentDebugImageIndex]?.name}
                  {captureSource === 'drone' && ' (Drone)'}
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
                  <Text style={[styles.navButton, currentDebugImageIndex === 0 && styles.navButtonDisabled]}>← Prev</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  onPress={() => setCurrentDebugImageIndex(Math.min(debugImages.length - 1, currentDebugImageIndex + 1))}
                  disabled={currentDebugImageIndex === debugImages.length - 1}
                >
                  <Text style={[styles.navButton, currentDebugImageIndex === debugImages.length - 1 && styles.navButtonDisabled]}>Next →</Text>
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
        {/* Enforcement Result Banner */}
        {enforcementResult && (
          <View style={[
            styles.enforcementBanner,
            enforcementResult.result === 'violation' ? styles.violationBanner :
            enforcementResult.result === 'approved' ? styles.approvedBanner :
            styles.unknownBanner
          ]}>
            <Text style={styles.enforcementResultText}>
              {enforcementResult.result === 'violation' ? '⚠️ VIOLATION' :
               enforcementResult.result === 'approved' ? '✅ APPROVED' :
               '⚠️ NEEDS REVIEW'}
            </Text>
            <Text style={styles.enforcementMessageText}>{enforcementResult.message}</Text>
            {enforcementResult.detected_zone && (
              <Text style={styles.enforcementZoneText}>
                Zone: {enforcementResult.detected_zone} ({enforcementResult.zone_type})
              </Text>
            )}
            {enforcementResult.dwell_minutes !== undefined && enforcementResult.limit_minutes !== undefined && (
              <Text style={styles.enforcementTimerText}>
                Time: {enforcementResult.dwell_minutes.toFixed(0)}m / {enforcementResult.limit_minutes}m
              </Text>
            )}
          </View>
        )}

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
          <>
            <TouchableOpacity
              style={[styles.button, isProcessing && styles.buttonDisabled]}
              onPress={handleTakePhoto}
              disabled={isProcessing}
            >
              <Text style={styles.buttonText}>
                {isProcessing && captureSource === 'phone' ? 'Processing...' : '📸 Capture & Analyze'}
              </Text>
            </TouchableOpacity>
            
            <TouchableOpacity
              style={[styles.button, styles.droneButton, isProcessing && styles.buttonDisabled]}
              onPress={handleCaptureDrone}
              disabled={isProcessing}
            >
              <Text style={styles.buttonText}>
                {isProcessing && captureSource === 'drone' ? 'Capturing...' : '🚁 Capture from Drone'}
              </Text>
            </TouchableOpacity>
            
            <TouchableOpacity
              style={[styles.button, styles.lastCaptureButton]}
              onPress={() => {
                setLastCaptureDebugIndex(0);
                setShowLastCaptureModal(true);
              }}
            >
              <Text style={styles.buttonText}>📋 View Last Capture</Text>
            </TouchableOpacity>
          </>
        ) : (
          <>
            <TouchableOpacity
              style={styles.button}
              onPress={handleCaptureNewPlate}
            >
              <Text style={styles.buttonText}>📷 Capture New Plate</Text>
            </TouchableOpacity>
            
            <TouchableOpacity
              style={[styles.button, styles.lastCaptureButton]}
              onPress={() => {
                setLastCaptureDebugIndex(0);
                setShowLastCaptureModal(true);
              }}
            >
              <Text style={styles.buttonText}>📋 View Last Capture</Text>
            </TouchableOpacity>
          </>
        )}
      </View>

      {isProcessing && (
        <View style={styles.loadingOverlay}>
          <ActivityIndicator size="large" color="#FFF" />
          <Text style={styles.loadingText}>
            {captureSource === 'drone' ? 'Capturing from drone stream...' : 'Processing...'}
          </Text>
        </View>
      )}
      
      {renderLastCaptureModal()}
    </View>
  );
}

const { width: screenWidth, height: screenHeight } = Dimensions.get('window');
const isWeb = Platform.OS === 'web';

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  text: {
    color: '#FFF',
    fontSize: 18,
    textAlign: 'center',
    marginTop: 100,
  },
  camera: {
    flex: 1,
    width: '100%',
    aspectRatio: isWeb ? undefined : undefined,
    minHeight: isWeb ? screenHeight * 0.5 : undefined,
    backgroundColor: '#000',
  },
  cameraOverlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'transparent',
  },
  focusGuide: {
    width: '85%',
    height: 120,
    borderWidth: 3,
    borderColor: 'rgba(0, 255, 0, 0.7)',
    borderRadius: 8,
  },
  debugSection: {
    flex: 1,
    backgroundColor: '#1a1a1a',
    justifyContent: 'center',
    alignItems: 'center',
    paddingVertical: 12,
    minHeight: isWeb ? screenHeight * 0.5 : undefined,
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
    width: '95%',
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
  navButtonDisabled: {
    color: '#555',
  },
  resultsPanel: {
    maxHeight: 280,
    backgroundColor: '#1a1a1a',
    padding: 14,
  },
  enforcementBanner: {
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
    borderWidth: 2,
  },
  violationBanner: {
    backgroundColor: '#4A0000',
    borderColor: '#FF3B30',
  },
  approvedBanner: {
    backgroundColor: '#003A00',
    borderColor: '#4CAF50',
  },
  unknownBanner: {
    backgroundColor: '#4A3500',
    borderColor: '#FFA500',
  },
  enforcementResultText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  enforcementMessageText: {
    color: '#FFF',
    fontSize: 13,
    marginBottom: 4,
  },
  enforcementZoneText: {
    color: '#CCC',
    fontSize: 11,
    marginTop: 4,
  },
  enforcementTimerText: {
    color: '#CCC',
    fontSize: 11,
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
    gap: 8,
  },
  button: {
    backgroundColor: '#007AFF',
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  droneButton: {
    backgroundColor: '#FF6B00',
  },
  lastCaptureButton: {
    backgroundColor: '#6B7280',
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
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    justifyContent: 'center',
    alignItems: 'center',
    gap: 12,
  },
  loadingText: {
    color: '#FFF',
    fontSize: 14,
  },
  modalContainer: {
    flex: 1,
    backgroundColor: '#000',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  modalTitle: {
    color: '#FFF',
    fontSize: 18,
    fontWeight: 'bold',
  },
  closeButton: {
    color: '#FFF',
    fontSize: 24,
    padding: 8,
  },
  modalContent: {
    flex: 1,
    padding: 14,
  },
  timestampText: {
    color: '#888',
    fontSize: 12,
    textAlign: 'center',
    marginBottom: 12,
  },
  modalDebugSection: {
    marginBottom: 16,
  },
  modalImageContainer: {
    width: '100%',
    height: 250,
    backgroundColor: '#1a1a1a',
    borderRadius: 8,
    borderWidth: 2,
    borderColor: '#00FF00',
    marginBottom: 16,
    overflow: 'hidden',
  },
  modalImage: {
    width: '100%',
    height: '100%',
    resizeMode: 'contain',
  },
  noCaptureContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  noCaptureText: {
    color: '#888',
    fontSize: 16,
  },
});