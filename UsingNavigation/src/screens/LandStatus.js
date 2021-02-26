import React, { Component } from 'react';
import {View, StyleSheet,Text} from 'react-native';

//once class tanımla sonra appnavigatorde screen ile tanımla
export default class LandStatusScreen extends Component {
    render() {
      return(
        <View style={styles.container}>
          <Text>Arazi Durumu!</Text>
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