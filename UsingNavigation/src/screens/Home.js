import React, { Component } from 'react';
import {View, StyleSheet,Text,Button,Image} from 'react-native';

class TitleLogo extends Component{
  render(){
    return(
      <Image 
      style={{width:38,height:38}}
      source={require('../assets/iconLeaf.png')}/>
    )
  }
}
export default class HomeScreen extends Component {
  static navigationOptions = {
    headerTitle: <TitleLogo/>,
    headerStyle: {
      backgroundColor: 'green'
    },
    headerTintColor: '#000',
    headerTitleStyle: {
      fontWeight: 'bold'
    }
  };
    render() {
        const { navigate } = this.props.navigation;
      return(
        <View style={styles.container}>
          <Text>Home!</Text>
          <Button
            onPress={() => navigate('About')}
            title="Learn More"
            color="#841584"
            accessibilityLabel="Learn more about TOAF"
            />
            <Button
            onPress={() => navigate('PlantDiseaseDetection')}
            title="PlantDiseaseDetection"
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