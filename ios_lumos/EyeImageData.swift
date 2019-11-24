//
//  EyeImageData.swift
//  lumos ios
//
//  Created by Johnathan Chen on 3/9/18.
//  Copyright © 2018 Shalin Shah. All rights reserved.
//

import Foundation
import UIKit

class EyeImageData {
    static let instance = EyeImageData()
    
    let defaults = UserDefaults.standard
    var selectedLeftOrRight: String {
        get {
            return defaults.object(forKey: "selectedLeftOrRight") as! String
        }
        set {
            defaults.set(newValue, forKey: "selectedLeftOrRight")
        }
    }

}
