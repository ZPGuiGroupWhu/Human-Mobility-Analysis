import React, { Component } from "react";
import './TableDrawer.css';
import { Table } from 'antd';
import Drawer from '@/components/drawer/Drawer';
import _ from 'lodash';

export default class TableDrawer extends Component {
  constructor(props) {
    super(props);
    this.Column = [{
      title: '数据',
      dataIndex: 'data',
      key: 'data',
      align: 'center',
    }];
    this.Data = [{
      key: '1',
      data: this.props.radar(),
    }, {
      key: '2',
      data: this.props.wordcloud(),
    }, {
      key: '3',
      data: this.props.violinplot(),
    }];
  }

  // 初始化数据
  changeData = () => {
    this.Data = [{
      key: '1',
      data: this.props.radar(),
    }, {
      key: '2',
      data: this.props.wordcloud(),
    }, {
      key: '3',
      data: this.props.violinplot(),
    }]
  };

  componentWillUpdate(prevProps, prevState) {
    this.changeData()
  }

  render() {
    return (
      <>
        <Drawer
          id={this.props.id}
          curId={this.props.curId}
          setCurId={this.props.setCurId}
          initVisible={false}
          type="right"
          width={this.props.rightWidth + 16}
          render={() => (
            <Table
              showHeader={false}
              scroll={{ y: 'calc(100vh - 70px)' }}
              pagination={false}
              dataSource={this.Data}
              columns={this.Column}
              bordered={true}
              width={this.props.rightWidth}
            />
          )}
        />
      </>
    );
  }

}