import React from 'react';
import { BrowserRouter as Router, Route, Switch } from 'react-router-dom'
import PagePredict from '@/project/predict/PagePredict';
import Mapgl from '@/project/predict/Mapgl';

export default function Predict(props) {
  return (
    <Router>
      <Switch>
        <Route exact path='/select/predict/'>
          <PagePredict />
        </Route>
        <Route exact path='/select/predict/gl'>
          <Mapgl />
        </Route>
      </Switch>
    </Router>
  )
}