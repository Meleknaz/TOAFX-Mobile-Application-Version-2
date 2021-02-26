import React, { Component } from 'react';

import { createAppContainer } from 'react-navigation';
import { createDrawerNavigator } from 'react-navigation-drawer';
import { createStackNavigator } from 'react-navigation-stack';
import HomeScreen from './screens/Home';
import WeatherScreen from './screens/Weather';
import WeatherAlertScreen from './screens/WeatherAlert';
import SatelliteImageScreen from './screens/SatelliteImage';
import LandStatusScreen from './screens/LandStatus';
import PlantDiseaseDetectionScreen from './screens/PlantDiseaseDetection';
import DeviceConnectionScreen from './screens/DeviceConnection';
import AboutScreen from './screens/About';


const HomeStack= createStackNavigator({
  Home: {
    screen: HomeScreen
  }
});

const WeatherStack= createStackNavigator({
    Weather: {
        screen: WeatherScreen
    }
});

const SatelliteImageStack= createStackNavigator({
    SatelliteImage: {
        screen: SatelliteImageScreen
      }
});

const WeatherAlertStack= createStackNavigator({
    WeatherAlert: {
        screen: WeatherAlertScreen
      }
});

const LandStatusStack= createStackNavigator({
    LandStatus: {
        screen: LandStatusScreen
      }
});

const PlantDiseaseDetectionStack= createStackNavigator({
    PlantDiseaseDetection: {
        screen: PlantDiseaseDetectionScreen
      }
});

const DeviceConnectionStack= createStackNavigator({
    DeviceConnection: {
        screen: DeviceConnectionScreen
      }
});

const AboutStack= createStackNavigator({
    About: {
        screen: AboutScreen
      }
});
 
 /* About: {
    screen: AboutScreen
  }
}, {
  initialRouteName: "Home",
  headerLayoutPreset: 'center',
  defaultNavigationOptions:{
    title: 'TOAFAPP',
    headerStyle: {
      backgroundColor: 'green'
    },
    headerTintColor: '#000',
    headerTitleStyle: {
      fontWeight: 'bold'
    }
  },
});*/

const Drawer = createDrawerNavigator({
    Home: HomeStack,
    Weather: WeatherStack,
    WeatherAlert: WeatherAlertStack,
    SatelliteImage: SatelliteImageStack,
    LandStatus: LandStatusStack,
    PlantDiseaseDetection: PlantDiseaseDetectionStack,
    DeviceConnection: DeviceConnectionStack,
    About: AboutStack
      
});

export default createAppContainer(Drawer);

