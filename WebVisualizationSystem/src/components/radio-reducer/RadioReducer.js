import React, { Component } from 'react'

class RadioReducer extends Component {
  constructor(props) {
    super(props);
    this.state = {
      curId: -1,
    }
  }

  getVisibility = (id) => {
    return (id === this.state.curId);
  }

  setVisibility = (id) => {
    this.setState(prev => {
      return {
        curId: (prev.curId === id) ? -1 : id ,
      }
    })
  }

  render() {
    return (
      React.Children.map(this.props.children, (elem, idx) => {
        if (!elem) {
          return null
        }

        return React.cloneElement(elem, {
          visibility: this.getVisibility(elem.props.id),
          setVisibility: this.setVisibility(elem.props.id),
        })
      })
    );
  }
}

export default RadioReducer;