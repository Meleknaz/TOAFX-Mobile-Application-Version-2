import React, { Component } from 'react';
import {View, StyleSheet,Text,Button} from 'react-native';

export default class AboutScreen extends Component {
   
    render() {
        const { navigate } = this.props.navigation;
      return(
        <View style={styles.container}>
          <Text>About!</Text>
          <Button
            onPress={() => navigate('Home')}
            title="Back Home Page"
            color="#841584"
            accessibilityLabel="back home page"
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