// 组件导入
import React from 'react';
import Main from '@/project/Main';
import TestPage from '@/network/TestPage';
// react-redux
import store from './app/store';
import { Provider } from 'react-redux';
// 样式导入
import './App.scss';

window.cancelList = {};

function App() {
  return (
    <Provider store={store}>
      <Main />
      <TestPage />
    </Provider>
  );
}

export default App;
