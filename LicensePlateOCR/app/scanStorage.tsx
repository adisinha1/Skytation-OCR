import AsyncStorage from '@react-native-async-storage/async-storage';

export interface ScanRecord {
  id: string;
  timestamp: number;
  licenseNumber: string | null;
  stateAbbreviation: string | null;
  image: string; // base64 image
}

const STORAGE_KEY = '@scan_history';
const MAX_SCANS = 5;

export const saveScan = async (scan: Omit<ScanRecord, 'id' | 'timestamp'>): Promise<void> => {
  try {
    const existingScans = await getScans();
    
    const newScan: ScanRecord = {
      id: Date.now().toString(),
      timestamp: Date.now(),
      ...scan,
    };
    
    // Add to beginning and keep only last 5
    const updatedScans = [newScan, ...existingScans].slice(0, MAX_SCANS);
    
    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(updatedScans));
  } catch (error) {
    console.error('Error saving scan:', error);
  }
};

export const getScans = async (): Promise<ScanRecord[]> => {
  try {
    const scansJson = await AsyncStorage.getItem(STORAGE_KEY);
    return scansJson ? JSON.parse(scansJson) : [];
  } catch (error) {
    console.error('Error getting scans:', error);
    return [];
  }
};

export const clearScans = async (): Promise<void> => {
  try {
    await AsyncStorage.removeItem(STORAGE_KEY);
  } catch (error) {
    console.error('Error clearing scans:', error);
  }
};