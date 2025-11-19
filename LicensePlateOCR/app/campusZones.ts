import AsyncStorage from '@react-native-async-storage/async-storage';

export interface CampusZone {
  id: string;
  name: string;
  code: string;
  latitude: number;
  longitude: number;
  radius: number; // in degrees (~0.0005 = 50 meters)
  createdAt: number;
}

const ZONES_STORAGE_KEY = '@campus_zones';

export const saveZone = async (zone: Omit<CampusZone, 'id' | 'createdAt'>): Promise<void> => {
  try {
    const zones = await getZones();
    const newZone: CampusZone = {
      ...zone,
      id: Date.now().toString(),
      createdAt: Date.now(),
    };
    const updatedZones = [...zones, newZone];
    await AsyncStorage.setItem(ZONES_STORAGE_KEY, JSON.stringify(updatedZones));
  } catch (error) {
    console.error('Error saving zone:', error);
  }
};

export const getZones = async (): Promise<CampusZone[]> => {
  try {
    const zonesJson = await AsyncStorage.getItem(ZONES_STORAGE_KEY);
    return zonesJson ? JSON.parse(zonesJson) : [];
  } catch (error) {
    console.error('Error getting zones:', error);
    return [];
  }
};

export const deleteZone = async (id: string): Promise<void> => {
  try {
    const zones = await getZones();
    const updatedZones = zones.filter(zone => zone.id !== id);
    await AsyncStorage.setItem(ZONES_STORAGE_KEY, JSON.stringify(updatedZones));
  } catch (error) {
    console.error('Error deleting zone:', error);
  }
};

export const clearAllZones = async (): Promise<void> => {
  try {
    await AsyncStorage.removeItem(ZONES_STORAGE_KEY);
  } catch (error) {
    console.error('Error clearing zones:', error);
  }
};

export const findZoneByCoordinates = (
  latitude: number,
  longitude: number,
  zones: CampusZone[]
): CampusZone | null => {
  for (const zone of zones) {
    const distance = Math.sqrt(
      Math.pow(latitude - zone.latitude, 2) + 
      Math.pow(longitude - zone.longitude, 2)
    );
    
    if (distance <= zone.radius) {
      return zone;
    }
  }
  return null;
};
