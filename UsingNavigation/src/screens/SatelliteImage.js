import React, { Component } from 'react';
import {View, StyleSheet,Text} from 'react-native';

//once class tanımla sonra appnavigatorde screen ile tanımla
export default class SatelliteImageScreen extends Component {
    render() {
      return(
        <View style={styles.container}>
          <Text>Satellite Image!</Text>
        </View>
      );
    }
  }
  const styles = StyleSheet.create({
    container: {
      flex:1,
      backgroundColor:'#F5FCFF',
      justifyContent: 'center',
      alignItems: 'center',
    }
  });  