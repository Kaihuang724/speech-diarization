import React, { Component } from 'react';
import logo from './logo.svg';
import './App.css';

class App extends Component {
  render() {
    return (
      <div className="App">
        <header className="App-header">
          <img src={logo} className="App-logo" alt="logo" />
          <h1 className="App-title">Welcome to React</h1>
        </header>
        <p className="App-intro">
            <a href="http://127.0.0.1:5000/learn"> Click here to start recording for training data </a>
            <a href="http://127.0.0.1:5000/test"> Click here to start recording for test data </a>
            <a href="http://127.0.0.1:5000/predict"> Click here to begin prediction </a>
        </p>
      </div>
    );
  }
}

export default App;
