import React, { Component } from 'react';
import logo from './logo.svg';
import './App.css';
import styled from 'styled-components';
import axios from 'axios'

const Wrapper = styled.section`
		padding: 4em;
`

class App extends Component {
	constructor(props) {
		super(props);
		this.state = {
			accuracy_score: 0,
			predictions: [],
			y_test: [],
			training_data: [],
			test_data: []
		}
	}

	trainSamples = async (name) => {
		try {
			const response = await axios.get("http://127.0.0.1:5000/learn", {
				params: {
					name: "kai"
				}
			})
			this.setState({
				training_data: [...this.state.training_data, {
					name: name,
					image_list: response.data.image_list
				}]
			})
		} catch(error) {
			console.error(error);
		}
	}

	testSamples = async (name) => {
		try {
			const response = await axios. post("http://127.0.0.1:5000/record_test", {
				name
			})
			console.log(response)
		} catch(error) {
			console.error(error);
		}
	}

	runPrediction = async () => {
		try {
			const response = await axios.get("http://127.0.0.1:5000/predict");
			this.setState({
				accuracy_score: response.data.accuracy_score,
				predictions: response.data.predictions,
				y_test: response.data.y_test
			})
		} catch (error) {
			console.error(error);
		}
	}

	render() {
		return (
			<Wrapper>
				<h1 className="App-title">Speech Diarization Test</h1>
				<button onClick={() => this.trainSamples(this.state.name)}> Record Training Data </button>
				<button onClick={() => this.testSamples(this.state.name)}> Record Test Data </button>
				<button onClick={this.runPrediction}> Run Prediction </button>
			</Wrapper>
		);
	}
}

export default App;
