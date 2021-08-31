import React, { Component } from 'react';
import "./Footer.scss";
import {Card, Col, Row ,Pagination,Popover} from 'antd';
class Footer extends Component {
  constructor(props) {
    super(props);
    this.pageSize=6;
    this.state = {
      currentPage:1,
      minValue: 0,
      maxValue: this.pageSize,
    }
  }
  onChange = (page) => {
    console.log(page);
    if (page <= 1) {
      this.setState({
        minValue: 0,
        maxValue: this.pageSize
      });
    } else {
      this.setState({
        minValue: (page-1) * this.pageSize,
        maxValue: page*this.pageSize
      });
    }
    this.setState({
      currentPage: page,
    });
  };
  onCardClic=(card)=>{
    console.log(card.target)
  }
  render() {
    let data = [
      { title: "Card title1", value: "content1" },
      { title: "Card title2", value: "content2" },
      { title: "Card title3", value: "content3" },
      { title: "Card title4", value: "content4" },
      { title: "Card title5", value: "content5" },
      { title: "Card title6", value: "content6" },
      { title: "Card title7", value: "content7" },
      { title: "Card title8", value: "content8" },
      { title: "Card title9", value: "content9" },
      { title: "Card title10", value: "content10" },
      { title: "Card title11", value: "content11" },
      { title: "Card title12", value: "content12" },
      { title: "Card title13", value: "content13" },
      { title: "Card title14", value: "content14" },
      { title: "Card title15", value: "content15" }
    ];
    return (
      <div className="select-footer-ctn">
      <Row gutter={16} style={{height:"90%"}}>
      {data &&
          data.length > 0 &&
          data.slice(this.state.minValue, this.state.maxValue).map(val => (
            <Col span={4}>
            <Popover content={
              <p>{val.value}</p>
            } title={val.title} trigger="click">
            <Card
              title={val.title}
              style={{ width: 100 }}
              hoverable={true}
              size="small"
            >
              <p>{val.value}</p>
            </Card>
            </Popover>
            </Col>
          ))}
      </Row>
        <Pagination style={{position: "absolute",left: "50%",top:"96%",transform:"translate(-50%, 0)"}}
          size="small" current={this.state.currentPage} onChange={this.onChange} total={data.length} showQuickJumper
          defaultPageSize={this.pageSize} showTotal={(total, range) => `${range[0]}-${range[1]} of ${total} items`}/>
      </div>
    );
  }
}

export default Footer;