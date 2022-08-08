import _ from "lodash";
import { useEffect, useState } from "react";


export const useSelectLogic = () => {
  const checkedOptions = [{ label: "双选", value: "double-select", disabled: false }];
  const [checkedValue, setCheckedValue] = useState([]);
  const doubleSelectLogic = () => {
    setCheckedValue((prev) => {
      const newState = [...prev];
      const removeItems = _.remove(newState, function (value) {
        return value === "double-select";
      });
      if (removeItems.length === 0) {
        newState.push("double-select");
      }
      return newState;
    });
  };
  const onCheckboxValueChange = (value) => {
    doubleSelectLogic();
  };
  useEffect(() => {
    const pressShift = (ev) => {
      if (ev.key === "Shift") {
        doubleSelectLogic();
      }
    };
    window.addEventListener("keyup", pressShift);
    return () => {
      window.removeEventListener("keyup", pressShift);
    };
  }, []);

  return { checkedOptions, checkedValue, setCheckedValue, onCheckboxValueChange };
};
