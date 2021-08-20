import React, { Component, createRef } from 'react';
import Plotly from 'plotly.js-dist-min';
import _ from "lodash";

export default class ViolinPlot extends Component{
    constructor(props) {
        super(props);
    };
    chartRef = createRef();
    unPack(rows, key) {
        return rows.map(function(row) { return row[key]; });
    }

    getViolinPlot = () => {
        const data = this.props.data;
        const option = this.props.option;
        const userData = [];
        _.forEach(data, function(item){
            userData.push(item)
        });

        const drawData = [{
            // width:1,
            // height:300,
            type: 'violin',
            x: this.unPack(userData, option),
            hoverinfo: 'x',
            hoverlabel:{
                bgcolor:'#FFC0CB',
                bordercolor: 'grey',
                font:{
                    family:'Microsoft Yahei',
                    size: 10,
                    color: 'black'
                }
            },
            hoveron: 'violins+kde',
            meanline:{
                visable: true,
                width: 2,
                color:'#000000'
            },
            box: {
                visible: true,
                width: 0.05,
                // fillcolor:'black',
                line:{
                    color: 'black',
                    width: 2
                }
            },
            spanmode: 'soft',
            line: {
                color: 'black',
                width: 1
            },
            fillcolor: '#87CEEB',
            opacity: 0.6,
            marker:{
                symbol:'diamond',
            },
            points: 'all',
            pointpos: 0,
            selectedpoints: [1],
            selected:{
                marker:{
                    color: '#FF0000',
                    size: 6,
                }
            },
            unselected:{
                marker:{
                    opacity:0,
                }
            },
            y0: "Total Bill"
        }];

        const layout = {
            paper_bgcolor: '#F5F5F5',
            plot_bgcolor: '#F5F5F5',
            xaxis: {
                zeroline: false
            },
            height:300,
            width:500,
            margin:{
                b:40,
                l:80,
                r:10,
                t:40
            }
        };

        //加载图表
        Plotly.newPlot(this.chartRef.current, drawData, layout);
        // resize
        window.onresize = Plotly.resize;
    };
    componentDidMount() {
        this.getViolinPlot();
    }
    componentDidUpdate(prevProps, prevState, snapshot) {
        this.getViolinPlot();
    }

    render() {
        return (
            <div
                className={'violinPlot'}
                ref={this.chartRef}
            ></div>
        );
    }
}



