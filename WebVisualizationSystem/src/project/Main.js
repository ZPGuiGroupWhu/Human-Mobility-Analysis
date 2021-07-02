// 组件导入
import React, { useState } from 'react';
import { BrowserRouter as Router, Redirect, Route, Switch } from 'react-router-dom'
import Layout from '@/components/layout/Layout';
import FunctionBar from '@/project/FunctionBar';
import PageAnalysis from '@/project/PageAnalysis';
import PageSelect from '@/project/PageSelect';
import PagePredict from '@/project/PagePredict';
// 自定义 Hook 导入
import { useResize } from '@/common/hooks/useResize';
// Context 对象导入
import { windowResize, drawerVisibility } from '@/context/mainContext';
// 图片
import { logo } from '@/icon'




function Main() {
  // 窗口 resize 监听
  const isResize = useResize(200);
  const initParams = {
    initCenter: '深圳市',
    initZoom: 12,
  }

  // 侧边栏 visibility
  const [leftDrawerVisible, setLeftDrawerVisible] = useState(false);
  const [rightDrawerVisible, setRightDrawerVisible] = useState(false);

  return (
    <windowResize.Provider value={isResize}>
      <drawerVisibility.Provider
        value={{
          leftDrawerVisible,
          rightDrawerVisible,
          setLeftDrawerVisible,
          setRightDrawerVisible
        }}
      >
        <Router>
          <Layout
            src={logo}
            title='轨迹目的地预测平台'
            imgHeight='60%'
          >
            {/* {
              <FunctionBar
                setLeftDrawerVisible={setLeftDrawerVisible}
                setRightDrawerVisible={setRightDrawerVisible}
              />
            } */}
            {
              <Switch>
                <Route exact path='/select'>
                  <PageSelect {...initParams} />
                </Route>
                <Route exact path='/select/analysis' render={() => <PageAnalysis {...initParams} />} />
                <Route exact path='/select/predict' render={() => <PagePredict {...initParams} />} />
                {/* 若均未匹配，重定向至首页 */}
                <Redirect to='/select' />
              </Switch>
            }
          </Layout>
        </Router>
      </drawerVisibility.Provider>
    </windowResize.Provider>
  );
}

export default Main;