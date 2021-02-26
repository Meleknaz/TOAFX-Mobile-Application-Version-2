import React, { Component } from 'react';
import {View, StyleSheet,Text,Button,goBack} from 'react-native';

//once class tanımla sonra appnavigatorde screen ile tanımla
export default class PlantDiseaseDetectionScreen extends Component {
  static navigationOptions = {
    title: 'TOAFAPP',
    alignItems: 'center'
  };
    render() {
      const { navigate, goBack } = this.props.navigation;
      return(
        <View style={styles.container}>
          <Text>Bitki hastalık tespiti!</Text>
          <Button
          title="go back"
            onPress={() => goBack()}
            color="#B00020"
            />
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